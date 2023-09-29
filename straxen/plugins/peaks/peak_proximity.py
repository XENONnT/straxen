import numpy as np
import numba
import strax
import straxen


export, __all__ = strax.exporter()


@export
class PeakProximity(strax.OverlapWindowPlugin):
    """Look for peaks around a peak to determine how many peaks are in proximity (in time) of a
    peak."""

    __version__ = "0.4.0"

    depends_on = ("peak_basics",)
    dtype = [
        ("n_competing", np.int32, "Number of nearby larger or slightly smaller peaks"),
        (
            "n_competing_left",
            np.int32,
            "Number of larger or slightly smaller peaks left of the main peak",
        ),
        (
            "t_to_prev_peak",
            np.int64,
            "Time between end of previous peak and start of this peak [ns]",
        ),
        ("t_to_next_peak", np.int64, "Time between end of this peak and start of next peak [ns]"),
        ("t_to_nearest_peak", np.int64, "Smaller of t_to_prev_peak and t_to_next_peak [ns]"),
    ] + strax.time_fields

    min_area_fraction = straxen.URLConfig(
        default=0.5,
        infer_type=False,
        help=(
            "The area of competing peaks must be at least "
            "this fraction of that of the considered peak"
        ),
    )

    nearby_window = straxen.URLConfig(
        default=int(1e7),
        infer_type=False,
        help="Peaks starting within this time window (on either side) in ns count as nearby.",
    )

    peak_max_proximity_time = straxen.URLConfig(
        default=int(1e8),
        infer_type=False,
        help="Maximum value for proximity values such as t_to_next_peak [ns]",
    )

    def get_window_size(self):
        return self.peak_max_proximity_time

    def compute(self, peaks):
        windows = strax.touching_windows(peaks, peaks, window=self.nearby_window)
        n_left, n_tot = self.find_n_competing(peaks, windows, fraction=self.min_area_fraction)

        t_to_prev_peak = np.ones(len(peaks), dtype=np.int64) * self.peak_max_proximity_time
        t_to_prev_peak[1:] = peaks["time"][1:] - peaks["endtime"][:-1]

        t_to_next_peak = t_to_prev_peak.copy()
        t_to_next_peak[:-1] = peaks["time"][1:] - peaks["endtime"][:-1]

        return dict(
            time=peaks["time"],
            endtime=strax.endtime(peaks),
            n_competing=n_tot,
            n_competing_left=n_left,
            t_to_prev_peak=t_to_prev_peak,
            t_to_next_peak=t_to_next_peak,
            t_to_nearest_peak=np.minimum(t_to_prev_peak, t_to_next_peak),
        )

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def find_n_competing(peaks, windows, fraction):
        n_left = np.zeros(len(peaks), dtype=np.int32)
        n_tot = n_left.copy()
        areas = peaks["area"]

        for i, peak in enumerate(peaks):
            left_i, right_i = windows[i]
            threshold = areas[i] * fraction
            n_left[i] = np.sum(areas[left_i:i] > threshold)
            n_tot[i] = n_left[i] + np.sum(areas[i + 1 : right_i] > threshold)

        return n_left, n_tot
