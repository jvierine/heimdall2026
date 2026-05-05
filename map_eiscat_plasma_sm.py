#!/usr/bin/env python3
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pyproj import Transformer
from spacepy.coordinates import Coords
from spacepy.time import Ticktock


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


def map_eiscat_points(mat_file, coord="SM"):
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

    coords = Coords(geo_points, "GEO", "car")
    coords.ticks = Ticktock(point_times, "UNX")
    with warnings.catch_warnings():
        warnings.simplefilter("once")
        mapped = coords.convert(coord, "car")

    return {
        "x": mapped.data[:, 0],
        "y": mapped.data[:, 1],
        "z": mapped.data[:, 2],
        "plasma": {key: value.T.reshape(-1) for key, value in plasma.items()},
        "coord": coord,
    }


def plot_parameter(mapped, parameter, output, dpi=200):
    plot_defs = {
        "ne": ("Electron density", "log10 ne (m$^{-3}$)", "plasma",
               lambda x: np.log10(np.where(x > 0, x, np.nan))),
        "Te": ("Electron temperature", "Te (K)", "turbo", lambda x: x),
        "Ti": ("Ion temperature", "Ti (K)", "turbo", lambda x: x),
        "vi": ("Ion velocity", "vi (m/s)", "seismic", lambda x: x),
    }
    title, label, cmap, transform = plot_defs[parameter]
    values = transform(mapped["plasma"][parameter])
    good = np.isfinite(values) & np.isfinite(mapped["x"]) & np.isfinite(mapped["y"])
    if parameter == "vi":
        vmax = np.nanpercentile(np.abs(values[good]), 98)
        vmin = -vmax
    else:
        vmin, vmax = np.nanpercentile(values[good], [2, 98])

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        mapped["x"][good],
        mapped["y"][good],
        c=values[good],
        s=6,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.scatter([0], [0], c="black", s=30, marker="o", label="Earth")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("%s X (Re)" % mapped["coord"])
    ax.set_ylabel("%s Y (Re)" % mapped["coord"])
    ax.set_title("BELLA %s in %s coordinates" % (title, mapped["coord"]))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print("saved %s" % output)
    return good


def main():
    parser = argparse.ArgumentParser(
        description="Plot BELLA EISCAT plasma parameters directly in SpacePy coordinates."
    )
    parser.add_argument("mat_file", nargs="?", default="bella_20260202.mat")
    parser.add_argument("--coord", default="SM", help="SpacePy coordinate system, e.g. SM, GSM, GEO.")
    parser.add_argument("-o", "--output-prefix", default="bella_20260202_direct_sm")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    mapped = map_eiscat_points(args.mat_file, coord=args.coord)
    mapped_finite = np.isfinite(mapped["x"]) & np.isfinite(mapped["y"])
    print("mapped %d/%d finite direct EISCAT points" %
          (np.count_nonzero(mapped_finite), len(mapped_finite)))
    print("median |%s Z|: %.3f Re" %
          (args.coord, np.nanmedian(np.abs(mapped["z"][mapped_finite]))))

    for parameter in ["ne", "Te", "Ti", "vi"]:
        good = plot_parameter(
            mapped,
            parameter,
            "%s_%s.png" % (args.output_prefix, parameter),
            dpi=args.dpi,
        )
        print("  %s finite plotted points: %d/%d" %
              (parameter, np.count_nonzero(good), len(good)))


if __name__ == "__main__":
    main()
