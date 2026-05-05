import numpy as np
import scipy.ndimage as ndi


DEFAULT_HIGHPASS_TIME_HOURS = 4.0
DEFAULT_HIGHPASS_RANGE_KM = 100.0


def centered_window_samples(width, spacing):
    if not np.isfinite(width) or width <= 0:
        return 1
    if not np.isfinite(spacing) or spacing <= 0:
        return 1
    nsamp = max(1, int(round(width/spacing)))
    if nsamp % 2 == 0:
        nsamp += 1
    return nsamp


def nanmean_boxcar(data, size):
    finite = np.isfinite(data)
    values = np.where(finite, data, 0.0)
    weights = finite.astype(float)
    smoothed_values = ndi.uniform_filter(values, size=size, mode="reflect")
    smoothed_weights = ndi.uniform_filter(weights, size=size, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        return smoothed_values/smoothed_weights


def highpass_plasma(plasma, t, range_km,
                    time_hours=DEFAULT_HIGHPASS_TIME_HOURS,
                    range_width_km=DEFAULT_HIGHPASS_RANGE_KM):
    dt_hours = np.nanmedian(np.diff(t))/3600.0
    dr_km = np.nanmedian(np.diff(range_km))
    time_window = centered_window_samples(time_hours, dt_hours)
    range_window = centered_window_samples(range_width_km, dr_km)
    size = (range_window, time_window)

    highpassed = {}
    for key, value in plasma.items():
        if key == "ne":
            filtered_value = np.log10(np.where(value > 0, value, np.nan))
        else:
            filtered_value = value
        highpassed[key] = filtered_value - nanmean_boxcar(filtered_value, size)

    return highpassed, {
        "time_hours": time_hours,
        "range_width_km": range_width_km,
        "time_window": time_window,
        "range_window": range_window,
        "dt_hours": dt_hours,
        "dr_km": dr_km,
    }


def add_highpass_arguments(parser):
    parser.add_argument(
        "--highpass",
        action="store_true",
        help="High-pass filter plasma data in range-time before plotting or mapping.",
    )
    parser.add_argument(
        "--highpass-time-hours",
        type=float,
        default=DEFAULT_HIGHPASS_TIME_HOURS,
        help="Centered high-pass smoothing window in time, in hours.",
    )
    parser.add_argument(
        "--highpass-range-km",
        type=float,
        default=DEFAULT_HIGHPASS_RANGE_KM,
        help="Centered high-pass smoothing window in range, in km.",
    )


def print_highpass_info(info):
    print(
        "high-pass filtering range-time data: "
        "%.2f h (%d samples) by %.1f km (%d gates)" %
        (
            info["time_hours"],
            info["time_window"],
            info["range_width_km"],
            info["range_window"],
        )
    )
