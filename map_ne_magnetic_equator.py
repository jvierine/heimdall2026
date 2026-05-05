#!/usr/bin/env python3
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pyproj import Transformer
from spacepy.coordinates import Coords
import spacepy.irbempy as irbempy
from spacepy.time import Ticktock

from bella_highpass import add_highpass_arguments, highpass_plasma, print_highpass_info

# Cool visualizations of https://eos.org/science-updates/great-mysteries-of-the-earths-magnetotail
# magnetopause KHI and other plasma structures that could map the EISCAT measurements
RE_KM = 6371.2
EISCAT_VHF_LAT = 69 + 35/60 + 11/3600
EISCAT_VHF_LON = 19 + 13/60 + 38/3600
EISCAT_VHF_ALT_KM = 0.15


def enu_to_ecef_matrix(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.array([
        [-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
        [ np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
        [ 0.0,          np.cos(lat),             np.sin(lat)],
    ])


def beam_points_geo_car(range_km, az_deg, el_deg):
    """Return EISCAT VHF beam points as GEO Cartesian coordinates in Re."""
    geodetic_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = geodetic_to_ecef.transform(
        EISCAT_VHF_LON,
        EISCAT_VHF_LAT,
        EISCAT_VHF_ALT_KM*1e3,
    )
    site_ecef_km = np.array([x0, y0, z0], dtype=float)/1e3

    az = np.deg2rad(float(az_deg))
    el = np.deg2rad(float(el_deg))
    range_km = np.asarray(range_km, dtype=float)

    east = range_km*np.cos(el)*np.sin(az)
    north = range_km*np.cos(el)*np.cos(az)
    up = range_km*np.sin(el)
    enu = np.vstack([east, north, up])

    ecef_km = site_ecef_km[:, None] + enu_to_ecef_matrix(EISCAT_VHF_LAT, EISCAT_VHF_LON) @ enu
    return ecef_km.T/RE_KM


def quiet_omni(npts):
    """Return quiet nominal solar-wind parameters for T96-style models."""
    omni = {
        "Kp": np.full(npts, 2.0),
        "Dst": np.zeros(npts),
        "dens": np.full(npts, 5.0),
        "velo": np.full(npts, 400.0),
        "Pdyn": np.full(npts, 2.0),
        "ByIMF": np.zeros(npts),
        "BzIMF": np.zeros(npts),
    }
    for key in ["G1", "G2", "G3", "W1", "W2", "W3", "W4", "W5", "W6"]:
        omni[key] = np.ones(npts)
    return omni


def map_to_magnetic_equator(
        mat_file,
        ext_mag="T96",
        output_coord="GSM",
        quiet_fallback=True,
        highpass=False,
        highpass_time_hours=4.0,
        highpass_range_km=100.0):
    d = sio.loadmat(mat_file, squeeze_me=True)
    t = np.asarray(d["t"], dtype=float)
    range_km = np.asarray(d["h"], dtype=float)
    az = np.asarray(d["az"], dtype=float)
    el = np.asarray(d["el"], dtype=float)
    plasma = {
        "ne": np.asarray(d["ne"], dtype=float),
        "Te": np.asarray(d["Te"], dtype=float),
        "Ti": np.asarray(d["Ti"], dtype=float),
        "vi": np.asarray(d["vi"], dtype=float),
    }
    highpass_info = None
    if highpass:
        plasma, highpass_info = highpass_plasma(
            plasma,
            t,
            range_km,
            time_hours=highpass_time_hours,
            range_width_km=highpass_range_km,
        )
        print_highpass_info(highpass_info)

    n_range = len(range_km)
    n_time = len(t)
    geo_points = np.empty((n_range*n_time, 3), dtype=float)
    point_times = np.repeat(t, n_range)

    for ti in range(n_time):
        geo_points[ti*n_range:(ti + 1)*n_range, :] = beam_points_geo_car(
            range_km,
            az[ti],
            el[ti],
        )

    ticks = Ticktock(point_times, "UNX")
    loci = Coords(geo_points, "GEO", "car", use_irbem=True)

    ext_mag = ext_mag.upper()
    omnivals = None
    if ext_mag in ["0", "OPQUIET"]:
        # Avoid requiring ~/.spacepy/data/omnidata.h5 for static/internal
        # field models. IRBEM still wants a mag-input array, but these values
        # are ignored for extMag=0 and OPQUIET.
        omnivals = quiet_omni(len(ticks))
    elif quiet_fallback:
        # SpacePy's Qin-Denton download URL is sometimes stale and this data
        # set is in 2026. Use nominal quiet parameters unless the user asks to
        # require a local OMNI cache.
        print("using quiet nominal OMNI parameters for %s" % ext_mag)
        omnivals = quiet_omni(len(ticks))

    with warnings.catch_warnings():
        warnings.simplefilter("once")
        try:
            mapped = irbempy.find_magequator(
                ticks,
                loci,
                extMag=ext_mag,
                omnivals=omnivals,
            )
        except FileNotFoundError as err:
            raise FileNotFoundError(
                "SpacePy could not find OMNI/Qin-Denton data. Either use the "
                "default quiet fallback, or install OMNI data manually for "
                "physical T96 driving parameters."
            ) from err

    equator_geo = mapped["loci"]
    equator_geo.ticks = ticks
    equator_xy = equator_geo.convert(output_coord, "car")

    return {
        "x": equator_xy.data[:, 0],
        "y": equator_xy.data[:, 1],
        "z": equator_xy.data[:, 2],
        "plasma": {key: value.T.reshape(-1) for key, value in plasma.items()},
        "bmin": mapped["Bmin"],
        "coord": output_coord,
        "ext_mag": ext_mag,
        "highpass": highpass,
        "highpass_info": highpass_info,
    }


def plot_parameter(mapped, parameter, values, output, dpi=200):
    highpass = mapped.get("highpass", False)
    plot_defs = {
        "ne": {
            "title": "Electron density high-pass" if highpass else "Electron density",
            "label": "high-pass log10 ne" if highpass else "log10 ne (m$^{-3}$)",
            "cmap": "plasma",
            "transform": lambda x: x if highpass else np.log10(np.where(x > 0, x, np.nan)),
        },
        "Te": {
            "title": "Electron temperature high-pass" if highpass else "Electron temperature",
            "label": "high-pass Te (K)" if highpass else "Te (K)",
            "cmap": "turbo",
            "transform": lambda x: x,
        },
        "Ti": {
            "title": "Ion temperature high-pass" if highpass else "Ion temperature",
            "label": "high-pass Ti (K)" if highpass else "Ti (K)",
            "cmap": "turbo",
            "transform": lambda x: x,
        },
        "vi": {
            "title": "Ion velocity high-pass" if highpass else "Ion velocity",
            "label": "high-pass vi (m/s)" if highpass else "vi (m/s)",
            "cmap": "seismic",
            "transform": lambda x: x,
        },
    }
    pdef = plot_defs[parameter]
    plot_values = pdef["transform"](values)
    good = np.isfinite(plot_values) & np.isfinite(mapped["x"]) & np.isfinite(mapped["y"])
    if parameter == "vi":
        vmax = np.nanpercentile(np.abs(plot_values[good]), 98)
        vmin = -vmax
    else:
        vmin, vmax = np.nanpercentile(plot_values[good], [2, 98])

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        mapped["x"][good],
        mapped["y"][good],
        c=plot_values[good],
        s=6,
        cmap=pdef["cmap"],
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.scatter([0], [0], c="black", s=30, marker="o", label="Earth")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("%s X at magnetic equator (Re)" % mapped["coord"])
    ax.set_ylabel("%s Y at magnetic equator (Re)" % mapped["coord"])
    ax.set_title("BELLA %s mapped along B to magnetic equator" % pdef["title"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(pdef["label"])
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print("saved %s" % output)
    return good


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trace EISCAT VHF BELLA measurement points along magnetic field "
            "lines to the magnetic equatorial plane and plot ne there."
        )
    )
    parser.add_argument("mat_file", nargs="?", default="bella_20260202.mat")
    parser.add_argument(
        "-o",
        "--output-prefix",
        default="bella_20260202_mageq_gsm",
        help="Output filename prefix. Parameter names are appended.",
    )
    parser.add_argument(
        "--ext-mag",
        default="T96",
        help="IRBEM external magnetic field model.",
    )
    parser.add_argument(
        "--require-omni",
        action="store_true",
        help="Require SpacePy OMNI/Qin-Denton data instead of using quiet nominal parameters.",
    )
    parser.add_argument(
        "--coord",
        default="GSM",
        help="Coordinate system for plotting the equatorial points, e.g. SM or GEO.",
    )
    add_highpass_arguments(parser)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    mapped = map_to_magnetic_equator(
        args.mat_file,
        ext_mag=args.ext_mag,
        output_coord=args.coord,
        quiet_fallback=not args.require_omni,
        highpass=args.highpass,
        highpass_time_hours=args.highpass_time_hours,
        highpass_range_km=args.highpass_range_km,
    )
    mapped_finite = np.isfinite(mapped["x"]) & np.isfinite(mapped["y"])
    print("mapped %d/%d finite magnetic-equator points" %
          (np.count_nonzero(mapped_finite), len(mapped_finite)))
    print("median |%s Z| at mapped equator: %.3f Re" %
          (args.coord, np.nanmedian(np.abs(mapped["z"][mapped_finite]))))

    for parameter in ["ne", "Te", "Ti", "vi"]:
        output_prefix = args.output_prefix
        if args.highpass and output_prefix == "bella_20260202_mageq_gsm":
            output_prefix = "%s_hp" % output_prefix
        output = "%s_%s.png" % (output_prefix, parameter)
        good = plot_parameter(
            mapped,
            parameter,
            mapped["plasma"][parameter],
            output,
            dpi=args.dpi,
        )
        print("  %s finite plotted points: %d/%d" %
              (parameter, np.count_nonzero(good), len(good)))


if __name__ == "__main__":
    main()
