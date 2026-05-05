#!/usr/bin/env python3
import argparse
import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from bella_highpass import add_highpass_arguments, highpass_plasma, print_highpass_info


def unix_to_datetime64(t):
    return np.array([dt.datetime.fromtimestamp(float(ti), dt.UTC) for ti in t])


def robust_limits(x, percentiles=(2, 98)):
    good = np.isfinite(x)
    if not np.any(good):
        return None, None
    return np.nanpercentile(x[good], percentiles)


def plot_panel(ax, time, range_km, values, title, label, cmap, vmin=None, vmax=None):
    mesh = ax.pcolormesh(
        time,
        range_km,
        values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_ylabel("Range (km)")
    ax.grid(color="white", alpha=0.25, linewidth=0.4)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(label)
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Plot BELLA ne, Ti, Te, and vi as range-time panels."
    )
    parser.add_argument(
        "mat_file",
        nargs="?",
        default="bella_20260202.mat",
        help="Input MATLAB file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="bella_20260202_range_time.png",
        help="Output image filename.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output figure DPI.",
    )
    add_highpass_arguments(parser)
    args = parser.parse_args()

    d = sio.loadmat(args.mat_file, squeeze_me=True)
    time = unix_to_datetime64(np.asarray(d["t"]))
    range_km = np.asarray(d["h"], dtype=float)

    plasma = {
        "ne": np.asarray(d["ne"], dtype=float),
        "Ti": np.asarray(d["Ti"], dtype=float),
        "Te": np.asarray(d["Te"], dtype=float),
        "vi": np.asarray(d["vi"], dtype=float),
    }
    if args.highpass:
        plasma, highpass_info = highpass_plasma(
            plasma,
            np.asarray(d["t"], dtype=float),
            range_km,
            time_hours=args.highpass_time_hours,
            range_width_km=args.highpass_range_km,
        )
        print_highpass_info(highpass_info)

    ne = plasma["ne"]
    ti = plasma["Ti"]
    te = plasma["Te"]
    vi = plasma["vi"]

    log_ne = ne if args.highpass else np.log10(np.where(ne > 0, ne, np.nan))
    ne_vmin, ne_vmax = robust_limits(log_ne)
    ti_vmin, ti_vmax = robust_limits(ti)
    te_vmin, te_vmax = robust_limits(te)
    vi_abs = 500#np.nanpercentile(np.abs(vi[np.isfinite(vi)]), 98)

    output = args.output
    if args.highpass and output == "bella_20260202_range_time.png":
        output = "bella_20260202_range_time_hp.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ne_title = "Electron density high-pass" if args.highpass else "Electron density"
    ti_title = "Ion temperature high-pass" if args.highpass else "Ion temperature"
    te_title = "Electron temperature high-pass" if args.highpass else "Electron temperature"
    vi_title = "Ion velocity high-pass" if args.highpass else "Ion velocity"
    ne_label = "high-pass log10 ne" if args.highpass else "log10 ne (m$^{-3}$)"
    ti_label = "high-pass Ti (K)" if args.highpass else "Ti (K)"
    te_label = "high-pass Te (K)" if args.highpass else "Te (K)"
    vi_label = "high-pass vi (m/s)" if args.highpass else "vi (m/s)"

    plot_panel(axes[0, 0], time, range_km, log_ne, ne_title, ne_label, "plasma", ne_vmin, ne_vmax)
    plot_panel(axes[0, 1], time, range_km, ti, ti_title, ti_label, "magma", ti_vmin, ti_vmax)
    plot_panel(axes[1, 0], time, range_km, te, te_title, te_label, "inferno", te_vmin, te_vmax)
    plot_panel(axes[1, 1], time, range_km, vi, vi_title, vi_label, "RdBu_r", -vi_abs, vi_abs)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (UTC)")
    for ax in axes.ravel():
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.suptitle(args.mat_file)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print("saved %s" % output)


if __name__ == "__main__":
    main()
