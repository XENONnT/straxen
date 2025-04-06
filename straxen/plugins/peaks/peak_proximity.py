import numpy as np
import numba
import strax
import straxen

from .peak_ambience import distance_in_xy, _quick_assign


export, __all__ = strax.exporter()


@export
class PeakProximity(strax.OverlapWindowPlugin):
    """Look for peaks around a peak to determine how many peaks are in proximity (in time) of a
    peak."""

    __version__ = "0.6.0"

    depends_on = ("peak_basics", "peak_positions")
    provides = "peak_proximity"
    dtype = [
        # when proximity_exponents is (1, -1, *), the unit is [/ms]
        ("proximity_score", np.float32, "Strength of proximity peaks in (time, space)"),
        (
            "n_competing_left",
            np.int32,
            "Number of larger or slightly smaller peaks left of the main peak",
        ),
        ("n_competing", np.int32, "Number of nearby larger or slightly smaller peaks"),
    ] + strax.time_fields

    proximity_type = straxen.URLConfig(
        default=2,
        type=(int, list),
        help="(List of) type(s) of peaks to consider for proximity score",
    )

    proximity_min_area_fraction = straxen.URLConfig(
        default=0.2,
        infer_type=False,
        help=(
            "The area of competing peaks must be at least "
            "this fraction of that of the considered peak"
        ),
    )

    proximity_window = straxen.URLConfig(
        default=int(1e7),
        infer_type=False,
        help="Peaks starting within this time window (on either side) in ns count as nearby.",
    )

    proximity_exponents = straxen.URLConfig(
        default=(1.0, -1.0, 1.0),
        type=(list, tuple),
        help="The exponent of (delta t, delta r) when calculating proximity score",
    )

    proximity_sigma = straxen.URLConfig(
        default=20.0,
        type=(int, float),
        help="The parameter of HalfCauchy, which is a function of S2 area",
    )

    def get_window_size(self):
        return 10 * self.proximity_window

    def compute(self, peaks):
        argsort = strax.stable_argsort(peaks["center_time"])
        _peaks = peaks[argsort].copy()
        result = np.zeros(len(peaks), self.dtype)
        _quick_assign(argsort, result, self.compute_proximity(peaks, _peaks))
        return result

    def compute_proximity(self, peaks, current_peak):
        result = np.zeros(len(current_peak), dtype=self.dtype)
        strax.set_nan_defaults(result)
        result["time"] = current_peak["time"]
        result["endtime"] = current_peak["endtime"]

        # only consider S1 and S2 peaks
        mask = np.isin(peaks["type"], self.proximity_type)

        roi = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi["time"] = current_peak["center_time"] - self.proximity_window
        roi["endtime"] = current_peak["center_time"].copy()
        result["proximity_score"] = self.peaks_proximity(
            current_peak,
            peaks[mask],
            strax.touching_windows(peaks[mask], roi),
            self.proximity_exponents,
            self.proximity_sigma,
            min_area_fraction=self.proximity_min_area_fraction,
        )

        roi = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi["time"] = current_peak["center_time"] - self.proximity_window
        roi["endtime"] = current_peak["center_time"] + self.proximity_window
        result["n_competing_left"], result["n_competing"] = self.find_n_competing(
            current_peak,
            peaks[mask],
            strax.touching_windows(peaks[mask], roi),
            fraction=self.proximity_min_area_fraction,
        )
        return result

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def find_n_competing(peaks, nearby_peaks, windows, fraction):
        n_left = np.zeros(len(peaks), dtype=np.int32)
        n = n_left.copy()
        nearby_areas = nearby_peaks["area"]
        areas = peaks["area"]

        dig = np.searchsorted(nearby_peaks["center_time"], peaks["center_time"])
        for i, peak in enumerate(peaks):
            left_i, right_i = windows[i]
            threshold = areas[i] * fraction
            n_left[i] = np.sum(nearby_areas[left_i : dig[i]] > threshold)
            n[i] = n_left[i] + np.sum(nearby_areas[dig[i] : right_i] > threshold)

        return n_left, n

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def peaks_proximity(
        peaks,
        pre_peaks,
        touching_windows,
        exponents,
        proximity_sigma,
        min_area_fraction=0.0,
    ):
        sum_array = np.zeros(len(peaks), np.float32)
        for p_i, suspicious_peak in enumerate(peaks):
            if suspicious_peak["area"] <= 0:
                continue
            indices = touching_windows[p_i]
            for idx in range(indices[0], indices[1]):
                creating_peak = pre_peaks[idx]
                dt = suspicious_peak["center_time"] - creating_peak["center_time"]
                dt /= 1e6  # convert from ns to ms
                if (dt <= 0) or (creating_peak["area"] <= 0):
                    continue
                if creating_peak["area"] < min_area_fraction * suspicious_peak["area"]:
                    continue
                dr = distance_in_xy(suspicious_peak, creating_peak)
                if np.isnan(dr):
                    continue
                if dr == 0:
                    # the peaks are at the same position
                    sum_array[p_i] = np.inf
                    continue
                r = creating_peak["area"] / suspicious_peak["area"]
                sigma = proximity_sigma * np.sqrt(
                    1 / creating_peak["area"] + 1 / suspicious_peak["area"]
                )
                pdf = half_cauchy_pdf(dr, sigma)
                score = r ** exponents[0] * dt ** exponents[1] * pdf ** exponents[2]
                sum_array[p_i] += score
        return sum_array


@numba.njit
def half_cauchy_pdf(x, scale=1.0):
    return np.where(
        x < 0.0,
        0.0,
        np.where(
            scale <= 0.0,
            2.0 / (np.pi * scale * (1 + (x / scale) ** 2)),
            np.nan,
        ),
    )
