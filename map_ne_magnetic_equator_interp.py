#!/usr/bin/env python3
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.interpolate as si
from pyproj import Transformer
from spacepy.coordinates import Coords
import spacepy.irbempy as irbempy
from spacepy.time import Ticktock

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


def map_to_magnetic_equator(mat_file, ext_mag="T96", output_coord="GSM", quiet_fallback=True):
    d = sio.loadmat(mat_file, squeeze_me=True)
    t = np.asarray(d["t"], dtype=float)
    range_km = np.asarray(d["h"], dtype=float)
    az = np.asarray(d["az"], dtype=float)
    el = np.asarray(d["el"], dtype=float)
    ne = np.asarray(d["ne"], dtype=float)

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
        "ne": ne.T.reshape(-1),
        "bmin": mapped["Bmin"],
        "coord": output_coord,
        "ext_mag": ext_mag,
    }


def interpolate_to_grid(x, y, values, xlim, ylim, nx, ny):
    grid_x = np.linspace(xlim[0], xlim[1], nx)
    grid_y = np.linspace(ylim[0], ylim[1], ny)
    gx, gy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack([x, y])

    linear = si.griddata(points, values, (gx, gy), method="linear")
    nearest = si.griddata(points, values, (gx, gy), method="nearest")
    grid_values = np.where(np.isfinite(linear), linear, nearest)
    return grid_x, grid_y, grid_values


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trace EISCAT VHF BELLA measurement points along magnetic field "
            "lines to the magnetic equatorial plane and plot ne there."
        )
    )
    parser.add_argument("mat_file", nargs="?", default="bella_20260202.mat")
    parser.add_argument("-o", "--output", default="bella_20260202_ne_mageq_sm_xy.png")
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
        default="SM",
        help="Coordinate system for plotting the equatorial points, e.g. SM or GEO.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--xmin", type=float, default=-40.0)
    parser.add_argument("--xmax", type=float, default=1.0)
    parser.add_argument("--ymin", type=float, default=-15.0)
    parser.add_argument("--ymax", type=float, default=-5.0)
    parser.add_argument("--nx", type=int, default=500)
    parser.add_argument("--ny", type=int, default=250)
    args = parser.parse_args()

    mapped = map_to_magnetic_equator(
        args.mat_file,
        ext_mag=args.ext_mag,
        output_coord=args.coord,
        quiet_fallback=not args.require_omni,
    )
    log_ne = np.log10(np.where(mapped["ne"] > 0, mapped["ne"], np.nan))
    good = np.isfinite(log_ne) & np.isfinite(mapped["x"]) & np.isfinite(mapped["y"])
    vmin, vmax = np.nanpercentile(log_ne[good], [2, 98])
    xlim = (args.xmin, args.xmax)
    ylim = (args.ymin, args.ymax)
    in_view = (
        good
        & (mapped["x"] >= xlim[0])
        & (mapped["x"] <= xlim[1])
        & (mapped["y"] >= ylim[0])
        & (mapped["y"] <= ylim[1])
    )

    grid_x, grid_y, grid_ne = interpolate_to_grid(
        mapped["x"][good],
        mapped["y"][good],
        log_ne[good],
        xlim,
        ylim,
        args.nx,
        args.ny,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    mesh = ax.pcolormesh(
        grid_x,
        grid_y,
        grid_ne,
        shading="auto",
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(mapped["x"][in_view], mapped["y"][in_view], s=1, c="black", alpha=0.15, linewidths=0)
    ax.scatter([0], [0], c="black", s=30, marker="o", label="Earth")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("%s X at magnetic equator (Re)" % args.coord)
    ax.set_ylabel("%s Y at magnetic equator (Re)" % args.coord)
    ax.set_title("BELLA ne mapped along B to magnetic equator")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("log10 ne (m$^{-3}$)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)

    print("mapped %d/%d finite ne points" % (np.count_nonzero(good), len(good)))
    print("interpolated %d points inside plot limits" % (np.count_nonzero(in_view)))
    print("%s X range of finite mapped points: %.3f to %.3f Re" %
          (args.coord, np.nanmin(mapped["x"][good]), np.nanmax(mapped["x"][good])))
    print("%s Y range of finite mapped points: %.3f to %.3f Re" %
          (args.coord, np.nanmin(mapped["y"][good]), np.nanmax(mapped["y"][good])))
    print("median |%s Z| at mapped equator: %.3f Re" % (args.coord, np.nanmedian(np.abs(mapped["z"][good]))))
    print("saved %s" % args.output)


if __name__ == "__main__":
    main()
