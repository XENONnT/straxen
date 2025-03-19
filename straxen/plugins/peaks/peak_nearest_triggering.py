import numpy as np
import numba
import strax
import straxen

from .peak_ambience import _quick_assign
from ..events import Events

export, __all__ = strax.exporter()


@export
class PeakNearestTriggering(Events):
    """Time difference and properties of the nearest triggering peaks in the left and right
    direction of peaks."""

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

    only_trigger_min_area = straxen.URLConfig(
        default=False,
        type=bool,
        track=True,
        help="Whether only require the triggering peak to have area larger than trigger_min_area",
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
        argsort = strax.stable_argsort(peaks, order="center_time")
        _peaks = strax.stable_sort(peaks, order="center_time")
        result = np.zeros(len(peaks), self.dtype)
        _quick_assign(argsort, result, self.compute_triggering(peaks, _peaks))
        return result

    def compute_triggering(self, peaks, current_peak):
        # sort peaks by center_time,
        # because later we will use center_time to find the nearest peak
        _peaks = strax.stable_sort(peaks, order="center_time")
        # only looking at triggering peaks
        if self.only_trigger_min_area:
            _is_triggering = _peaks["area"] > self.trigger_min_area
        else:
            _is_triggering = self._is_triggering(_peaks)
        _peaks = _peaks[_is_triggering]
        # init result
        result = np.zeros(len(current_peak), self.dtype)
        strax.set_nan_defaults(result)

        # use center_time as the anchor of things
        things = np.zeros(len(_peaks), dtype=strax.time_fields)
        things["time"] = _peaks["center_time"]
        things["endtime"] = _peaks["center_time"]
        # also se center_time as the anchor of containers
        containers = np.zeros(len(current_peak), dtype=strax.time_fields)
        containers["time"] = current_peak["center_time"] - self.shadow_time_window_backward
        containers["endtime"] = current_peak["center_time"] + self.shadow_time_window_backward

        # find indices of the nearest peaks in left and right direction
        split_peaks = strax.touching_windows(things, containers)
        left_indices, right_indices = self.nearest_indices(
            current_peak["center_time"],
            _peaks["center_time"],
            split_peaks,
        )
        # assign fields
        result["left_dtime"] = np.where(
            left_indices != -1,
            current_peak["center_time"] - _peaks["center_time"][left_indices],
            self.shadow_time_window_backward,
        )
        result["right_dtime"] = np.where(
            right_indices != -1,
            _peaks["center_time"][right_indices] - current_peak["center_time"],
            self.shadow_time_window_backward,
        )
        for field in ["time", "endtime", "center_time", "type", "n_competing", "area"]:
            result["left_" + field] = np.where(
                left_indices != -1, _peaks[field][left_indices], result["left_" + field]
            )
            result["right_" + field] = np.where(
                right_indices != -1, _peaks[field][right_indices], result["right_" + field]
            )
        result["time"] = current_peak["time"]
        result["endtime"] = strax.endtime(current_peak)
        for direction in ["left", "right"]:
            assert np.all(result[f"{direction}_dtime"] > 0), f"{direction}_dtime should be positive"
        return result

    @staticmethod
    @numba.njit
    def nearest_indices(reference_time, nearest_time, touching_windows):
        """Find the nearest indices in the left and right direction."""
        left_indices = np.full(len(reference_time), -1, dtype=np.int64)
        right_indices = np.full(len(reference_time), -1, dtype=np.int64)
        for r_i, r in enumerate(reference_time):
            indices = touching_windows[r_i]
            if indices[0] == indices[1]:
                continue
            center_time = nearest_time[indices[0] : indices[1]]
            left_dtime = r - center_time[0]
            right_dtime = center_time[-1] - r
            # make sure that the dtime is positive
            for p_i, t in enumerate(center_time):
                if t < r and r - t <= left_dtime:
                    left_dtime = r - t
                    left_indices[r_i] = indices[0] + p_i
                elif t > r and t - r <= right_dtime:
                    right_dtime = t - r
                    right_indices[r_i] = indices[0] + p_i
        return left_indices, right_indices
