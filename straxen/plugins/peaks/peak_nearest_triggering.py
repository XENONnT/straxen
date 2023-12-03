import numpy as np
import numba
import strax
import straxen
from .peak_ambience import _quick_assign
from ..events import Events

export, __all__ = strax.exporter()


@export
class PeakNearestTriggering(Events):
    """Time difference and properties of the nearest triggering peaks."""

    __version__ = "0.0.0"
    depends_on = ("peak_basics", "peak_proximity")
    provides = "peak_nearest_triggering"
    data_kind = "peaks"
    save_when = strax.SaveWhen.EXPLICIT

    shadow_time_window_backward = straxen.URLConfig(
        default=int(1e9),
        type=int,
        track=True,
        help="Search for peaks casting time & position shadow in this time window [ns]",
    )

    def infer_dtype(self):
        dtype = []
        common_descr = "of the nearest triggering peak on the"
        for direction in ["left", "right"]:
            dtype += [
                (
                    (f"time difference {common_descr} {direction} [ns]", f"{direction}_dtime"),
                    np.int64,
                ),
                ((f"time {common_descr} {direction} [ns]", f"{direction}_time"), np.int64),
                ((f"endtime {common_descr} {direction} [ns]", f"{direction}_endtime"), np.int64),
                (
                    (f"center_time {common_descr} {direction} [ns]", f"{direction}_center_time"),
                    np.int64,
                ),
                ((f"type {common_descr} {direction}", f"{direction}_type"), np.int8),
                ((f"n_competing {common_descr} {direction}", f"{direction}_n_competing"), np.int32),
                ((f"area {common_descr} {direction} [PE]", f"{direction}_area"), np.float32),
            ]
        dtype += strax.time_fields
        return dtype

    def get_window_size(self):
        # This method is required by the OverlapWindowPlugin class
        return 10 * self.shadow_time_window_backward

    def compute(self, peaks):
        result = self.compute_triggering(peaks, peaks)
        return result

    def compute_triggering(self, peaks, current_peak):
        _is_triggering = self._is_triggering(peaks)

        roi_triggering = np.zeros(len(current_peak), dtype=strax.time_fields)

        result = np.zeros(len(current_peak), self.dtype)
        straxen.EventBasics.set_nan_defaults(result)
        for direction, reference in zip(["right", "left"], ["endtime", "time"]):
            _result = result.copy()
            _result[f"{direction}_dtime"] = self.shadow_time_window_backward
            argsort = np.argsort(current_peak[reference], kind="mergesort")
            _current_peak = np.sort(current_peak, order=reference)
            # the calculation of the time difference seems weird,
            # but it is following the functionality of strax.find_peak_groups and strax.find_peaks
            # https://github.com/AxFoundation/strax/blob/21cc96e011b0e4099138979791f34e8b1addedb7/strax/processing/peak_building.py#L102
            # record the time difference to the nearest previous peak
            if direction == "left":
                roi_triggering["time"] = _current_peak[reference] - self.shadow_time_window_backward
                roi_triggering["endtime"] = _current_peak[reference].copy()
                split_peaks = strax.touching_windows(peaks[_is_triggering], roi_triggering)
                indices = self.peaks_triggering_indices(
                    direction, _current_peak["time"], peaks["endtime"][_is_triggering], split_peaks
                )
                _result[f"{direction}_dtime"] = np.where(
                    indices != -1,
                    _current_peak["time"] - peaks["endtime"][_is_triggering][indices],
                    self.shadow_time_window_backward,
                )
            elif direction == "right":
                roi_triggering["time"] = _current_peak[reference].copy()
                roi_triggering["endtime"] = (
                    _current_peak[reference] + self.shadow_time_window_backward
                )
                split_peaks = strax.touching_windows(peaks[_is_triggering], roi_triggering)
                indices = self.peaks_triggering_indices(
                    direction, _current_peak["endtime"], peaks["time"][_is_triggering], split_peaks
                )
                _result[f"{direction}_dtime"] = np.where(
                    indices != -1,
                    peaks["time"][_is_triggering][indices] - _current_peak["endtime"],
                    self.shadow_time_window_backward,
                )
            for field in ["time", "endtime", "center_time", "type", "n_competing", "area"]:
                _result[direction + "_" + field] = peaks[field][_is_triggering][indices]
            # When reference is 'endtime', the current_peak[reference] is not sorted,
            # so the _quick_assign will assign the _result to correct peaks.
            # So direction must be first right then left.
            _quick_assign(argsort, result, _result)
        result["time"] = current_peak["time"]
        result["endtime"] = strax.endtime(current_peak)
        return result

    @staticmethod
    @numba.njit
    def peaks_triggering_indices(direction, reference, triggering, touching_windows):
        indices = np.ones_like(reference) * -1
        for p_i in range(len(reference)):
            if touching_windows[p_i, 1] - touching_windows[p_i, 0] == 0:
                continue
            if direction == "left":
                indices[p_i] = touching_windows[p_i, 0] + np.argmax(
                    triggering[touching_windows[p_i, 0] : touching_windows[p_i, 1]]
                )
            elif direction == "right":
                indices[p_i] = touching_windows[p_i, 0] + np.argmin(
                    triggering[touching_windows[p_i, 0] : touching_windows[p_i, 1]]
                )
        return indices
