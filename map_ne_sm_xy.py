#!/usr/bin/env python3
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pyproj import Transformer
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

from bella_highpass import add_highpass_arguments, highpass_plasma, print_highpass_info


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


def beam_points_geo_car(range_km, az_deg, el_deg, site_lat, site_lon, site_alt_km):
    """Return beam points as GEO Cartesian coordinates in Earth radii."""
    geodetic_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = geodetic_to_ecef.transform(site_lon, site_lat, site_alt_km*1e3)
    site_ecef_km = np.array([x0, y0, z0], dtype=float)/1e3

    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    range_km = np.asarray(range_km, dtype=float)

    east = range_km*np.cos(el)*np.sin(az)
    north = range_km*np.cos(el)*np.cos(az)
    up = range_km*np.sin(el)
    enu = np.vstack([east, north, up])

    rot = enu_to_ecef_matrix(site_lat, site_lon)
    ecef_km = site_ecef_km[:, None] + rot @ enu
    return (ecef_km.T)/RE_KM


def map_to_spacepy_xy(
        mat_file,
        coord_system="SM",
        highpass=False,
        highpass_time_hours=4.0,
        highpass_range_km=100.0):
    d = sio.loadmat(mat_file, squeeze_me=True)
    t = np.asarray(d["t"], dtype=float)
    range_km = np.asarray(d["h"], dtype=float)
    plasma = {"ne": np.asarray(d["ne"], dtype=float)}
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

    az = np.asarray(d["az"], dtype=float)
    el = np.asarray(d["el"], dtype=float)

    n_range = len(range_km)
    n_time = len(t)
    geo_points = np.empty((n_range*n_time, 3), dtype=float)
    point_times = np.repeat(t, n_range)

    for ti in range(n_time):
        start = ti*n_range
        stop = start + n_range
        geo_points[start:stop, :] = beam_points_geo_car(
            range_km,
            az[ti],
            el[ti],
            EISCAT_VHF_LAT,
            EISCAT_VHF_LON,
            EISCAT_VHF_ALT_KM,
        )

    coords = Coords(geo_points, "GEO", "car")
    coords.ticks = Ticktock(point_times, "UNX")
    with warnings.catch_warnings():
        warnings.simplefilter("once")
        mapped = coords.convert(coord_system, "car")

    values = plasma["ne"].T.reshape(-1)
    return mapped.data[:, 0], mapped.data[:, 1], values, highpass_info


def main():
    parser = argparse.ArgumentParser(
        description="Map BELLA electron density from EISCAT VHF beam range gates into SpacePy coordinates."
    )
    parser.add_argument("mat_file", nargs="?", default="bella_20260202.mat")
    parser.add_argument("-o", "--output", default="bella_20260202_ne_sm_xy.png")
    parser.add_argument(
        "--coord",
        default="SM",
        help="SpacePy output coordinate system. SpacePy supports SM, GSM, GSE, GEO, MAG, etc. It does not define ISM.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    add_highpass_arguments(parser)
    args = parser.parse_args()

    x, y, ne, _ = map_to_spacepy_xy(
        args.mat_file,
        coord_system=args.coord,
        highpass=args.highpass,
        highpass_time_hours=args.highpass_time_hours,
        highpass_range_km=args.highpass_range_km,
    )
    log_ne = ne if args.highpass else np.log10(np.where(ne > 0, ne, np.nan))
    good = np.isfinite(log_ne)
    vmin, vmax = np.nanpercentile(log_ne[good], [2, 98])

    output = args.output
    if args.highpass and output == "bella_20260202_ne_sm_xy.png":
        output = "bella_20260202_ne_sm_xy_hp.png"

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        x[good],
        y[good],
        c=log_ne[good],
        s=6,
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("%s X (Re)" % args.coord)
    ax.set_ylabel("%s Y (Re)" % args.coord)
    title = "BELLA electron density mapped to %s X-Y" % args.coord
    if args.highpass:
        title = "BELLA high-pass electron density mapped to %s X-Y" % args.coord
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("high-pass log10 ne" if args.highpass else "log10 ne (m$^{-3}$)")
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print("saved %s" % output)


if __name__ == "__main__":
    main()
