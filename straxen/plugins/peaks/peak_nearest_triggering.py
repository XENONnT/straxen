import numpy as np
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
        argsort = np.argsort(peaks["center_time"], kind="mergesort")
        _peaks = np.sort(peaks, order="center_time")
        result = np.zeros(len(peaks), self.dtype)
        _quick_assign(argsort, result, self.compute_triggering(peaks, _peaks))
        return result

    def compute_triggering(self, peaks, current_peak):
        _is_triggering = self._is_triggering(peaks)

        roi_triggering = np.zeros(len(current_peak), dtype=strax.time_fields)

        result = np.zeros(len(current_peak), self.dtype)
        straxen.EventBasics.set_nan_defaults(result)
        for direction in ["left", "right"]:
            result[f"{direction}_dtime"] = self.shadow_time_window_backward
            if direction == "left":
                roi_triggering["time"] = (
                    current_peak["center_time"] - self.shadow_time_window_backward
                )
                roi_triggering["endtime"] = current_peak["center_time"].copy()
                split_peaks = strax.touching_windows(peaks[_is_triggering], roi_triggering)
                indices = np.clip(split_peaks[:, 1] - 1, 0, _is_triggering.sum() - 1)
                result[f"{direction}_dtime"] = np.where(
                    (split_peaks[:, 0] - split_peaks[:, 1] != 0) & (split_peaks[:, 1] != 0),
                    current_peak["center_time"] - peaks["center_time"][_is_triggering][indices],
                    self.shadow_time_window_backward,
                )
            elif direction == "right":
                # looking for peaks right to the current peaks
                roi_triggering["time"] = current_peak["center_time"].copy()
                roi_triggering["endtime"] = (
                    current_peak["center_time"] + self.shadow_time_window_backward
                )
                split_peaks = strax.touching_windows(peaks[_is_triggering], roi_triggering)
                indices = np.clip(split_peaks[:, 0], 0, _is_triggering.sum() - 1)
                result[f"{direction}_dtime"] = np.where(
                    (split_peaks[:, 0] - split_peaks[:, 1] != 0)
                    & (split_peaks[:, 0] != _is_triggering.sum()),
                    peaks["center_time"][_is_triggering][indices] - current_peak["center_time"],
                    self.shadow_time_window_backward,
                )
            for field in ["time", "endtime", "center_time", "type", "n_competing", "area"]:
                result[direction + "_" + field] = peaks[field][_is_triggering][indices]
        result["time"] = current_peak["time"]
        result["endtime"] = strax.endtime(current_peak)
        return result
