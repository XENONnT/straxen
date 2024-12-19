import numpy as np
import numba
import strax
import straxen


export, __all__ = strax.exporter()


@export
class PositionShadow(strax.CutPlugin):
    """This cut reject events within large position correlation to previous peak casting shadow. It
    is mainly against isolated S2.

    References:
        * v0.3.7: xenon:xenonnt:ac:prediction:shadow_ambience
        * v0.3.8: xenon:xenonnt:ac:ybe_suppress

    """

    depends_on = ("event_basics", "event_shadow")
    provides = "cut_position_shadow"
    cut_name = "cut_position_shadow"
    cut_description = (
        "Position shadow cut to reject events with strong position correlation with previous S2"
    )
    __version__ = "0.3.9"

    position_shadow_s2_area_limit = straxen.URLConfig(
        default=5e4,
        type=float,
        track=True,
        help="Max S2 area to implement this cut, LowER analysis upper limit [PE]",
    )

    position_shadow_ellipse_parameters_mode = straxen.URLConfig(
        default={
            "bkg": [-0.969, -2.485, 0.564, 0.766, 1.798],  # Used by bkg and ted runs
            "ar37": [-0.969, -2.485, 0.564, 0.766, 1.798],
            "radon": [-0.895, -2.226, 0.713, 0.818, 1.370],
            "radon_hev": [-0.873, -2.192, 0.624, 0.795, 1.504],
            "ambe": [-0.911, -1.947, 0.821, 1.047, 0.734],
            "kr83m": [-0.862, -2.531, 0.652, 0.779, 1.922],
            "ybe": [-0.961, -2.344, 0.796, 0.885, 1.534],
            "rn222": [-0.940, -2.537, 0.719, 0.829, 1.612],
        },
        type=dict,
        track=True,
        help="cut parameters for the ellipse curve in different run mode",
    )

    position_shadow_straight_parameters_mode = straxen.URLConfig(
        default={
            "bkg": [-0.947, -4.231],  # Used by bkg and ted runs
            "ar37": [-0.947, -4.231],
            "radon": [-0.975, -3.388],
            "radon_hev": [-0.868, -3.110],
            "ambe": [-0.430, -1.993],
            "kr83m": [-1.022, -4.195],
            "ybe": [-0.801, -2.670],
            "rn222": [-0.929, -4.009],
        },
        type=dict,
        track=True,
        help="cut parameters for the straight line in different run mode",
    )

    run_mode = straxen.URLConfig(
        default="run_mode://plugin.run_id",
        help="Run mode to be used for the cut. It can affect the parameters of the cut.",
    )

    def setup(self):
        self.s2_area_limit = self.position_shadow_s2_area_limit
        self.ellipse_parameters = self.position_shadow_ellipse_parameters_mode[self.run_mode]
        self.straight_parameters = self.position_shadow_straight_parameters_mode[self.run_mode]

    @staticmethod
    @np.errstate(invalid="ignore")
    def prob_cut_line(log10_pdf, x0, y0, a, b, rlimit, pstraight):
        top = 0
        # The top line in 2D histogram
        topline = np.full(len(log10_pdf), top)

        # Draw a half ellipse
        halfellipse = np.where(
            np.abs(log10_pdf - x0) < a * rlimit,
            y0 - np.sqrt(b**2 * (rlimit**2 - (log10_pdf - x0) ** 2 / a**2)),
            top,
        )

        # A horizontal tangent line at the bottom of ellipse
        botline = np.where(log10_pdf > x0, y0 - b * rlimit, top)

        # A straight line with slope
        straight = np.polyval(pstraight, log10_pdf)

        # The lowest of 4 lines
        line = np.min(np.vstack([topline, halfellipse, botline, straight]), axis=0)
        return line

    @np.errstate(divide="ignore")
    def cut_by(self, events):
        log10_time_shadow = np.log10(
            events["s2_shadow_s2_position_shadow"] / events["s2_pdf_s2_position_shadow"]
        )
        log10_pdf = np.log10(events["s2_pdf_s2_position_shadow"])
        mask = ~(
            log10_time_shadow
            > self.prob_cut_line(log10_pdf, *self.ellipse_parameters, self.straight_parameters)
        )
        mask |= events["s2_area"] > self.s2_area_limit
        return mask


@export
class PeakPositionShadowCut(PositionShadow):
    """A peak-level cut This cut is defined in the position shadow space.

    It is only against S2 peaks.

    """

    depends_on = ("peak_basics", "peak_shadow")
    provides = "cut_position_shadow_peak"
    cut_name = "cut_position_shadow_peak"
    data_kind = "peaks"
    cut_description = "peak-level position shadow cut"
    __version__ = "0.0.0"

    def cut_by(self, peaks):
        events = np.zeros(
            len(peaks),
            dtype=[
                ("s2_area", np.float32),
                ("s2_shadow_s2_position_shadow", np.float32),
                ("s2_pdf_s2_position_shadow", np.float32),
            ],
        )
        events["s2_area"] = peaks["area"]
        events["s2_shadow_s2_position_shadow"] = peaks["shadow_s2_position_shadow"]
        events["s2_pdf_s2_position_shadow"] = peaks["pdf_s2_position_shadow"]
        mask = super().cut_by(events)
        mask |= peaks["type"] != 2
        return mask


@export
class PeakProximity(strax.OverlapWindowPlugin):
    """Look for peaks around a peak to determine how many peaks are in proximity (in time) of a
    peak."""

    __version__ = "0.4.0"

    depends_on = ("peak_basics", "peak_shadow", "peak_se_score", "cut_position_shadow_peak")
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
        mask_good = peaks["cut_position_shadow_peak"]
        mask_good &= peaks["se_score"] > 0.1

        for i, peak in enumerate(peaks):
            left_i, right_i = windows[i]
            threshold = areas[i] * fraction
            n_left[i] = np.sum((areas[left_i:i] > threshold) & mask_good[left_i:i])
            n_tot[i] = n_left[i] + np.sum(
                (areas[i + 1 : right_i] > threshold) & mask_good[i + 1 : right_i]
            )

        return n_left, n_tot
