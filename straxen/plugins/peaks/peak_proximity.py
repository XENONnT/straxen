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
        ("proximity_score", np.float32, "Strength of proximity peaks in (time, space) [/ns/cm]"),
        (
            "n_competing_left",
            np.int32,
            "Number of larger or slightly smaller peaks left of the main peak",
        ),
        ("n_competing", np.int32, "Number of nearby larger or slightly smaller peaks"),
    ] + strax.time_fields

    proximity_max_peak = straxen.URLConfig(
        default=1e4,
        type=(int, float),
        help="Exclude larger peaks than this from proximity peak [PE]",
    )

    proximity_min_area_fraction = straxen.URLConfig(
        default=0.5,
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
        default=(-1.0, -1.0),
        type=(list, tuple),
        help="The exponent of (delta t, delta r) when calculating proximity score",
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

        roi = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi["time"] = current_peak["center_time"] - self.proximity_window
        roi["endtime"] = current_peak["center_time"].copy()
        mask = peaks["type"] == 2
        mask &= peaks["area"] < self.proximity_max_peak
        result["proximity_score"] = self.peaks_proximity(
            current_peak,
            peaks[mask],
            strax.touching_windows(peaks[mask], roi),
            self.proximity_exponents,
            min_area_fraction=self.proximity_min_area_fraction,
        )

        roi = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi["time"] = current_peak["center_time"] - self.proximity_window
        roi["endtime"] = current_peak["center_time"] + self.proximity_window
        # only consider S1 and S2 peaks
        mask = np.isin(peaks["type"], [1, 2])
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
        n_tot = n_left.copy()
        areas_in_roi = nearby_peaks["area"]
        areas = peaks["area"]

        dig = np.searchsorted(nearby_peaks["center_time"], peaks["center_time"])
        for i, peak in enumerate(peaks):
            left_i, right_i = windows[i]
            threshold = areas[i] * fraction
            n_left[i] = np.sum(areas_in_roi[left_i : dig[i]] > threshold)
            n_tot[i] = n_left[i] + np.sum(areas_in_roi[dig[i] : right_i] > threshold)

        return n_left, n_tot

    @staticmethod
    @numba.njit(nopython=True, nogil=True, cache=True)
    def peaks_proximity(
        peaks,
        pre_peaks,
        touching_windows,
        exponents,
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
                if (dt <= 0) or (creating_peak["area"] <= 0):
                    continue
                if creating_peak["area"] < min_area_fraction * suspicious_peak["area"]:
                    continue
                r = distance_in_xy(suspicious_peak, creating_peak)
                if np.isnan(r):
                    continue
                if r == 0:
                    # the peaks are at the same position
                    sum_array[p_i] = np.inf
                    continue
                score = creating_peak["area"] * dt ** exponents[0] * r ** exponents[1]
                score /= suspicious_peak["area"]
                sum_array[p_i] += score
        return sum_array
