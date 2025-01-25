import strax
import numpy as np
import straxen

export, __all__ = strax.exporter()


@export
class Events(strax.OverlapWindowPlugin):
    """Plugin which defines an "event" in our TPC.

    An event is defined by peak(s) in fixed range of time around a peak
    which satisfies certain conditions:
        1. The triggering peak must have a certain area.
        2. The triggering peak must have less than
           "trigger_max_competing" peaks. (A competing peak must have a
           certain area fraction of the triggering peak and must be in a
           window close to the main peak)

    Note:
        The time range which defines an event gets chopped at the chunk
        boundaries. This happens at invalid boundaries of the

    """

    __version__ = "0.1.1"

    depends_on = (
        "peak_basics",
        "peak_proximity",
        "peak_shadow",
        "peak_se_score",
    )
    provides = "events"
    data_kind = "events"

    save_when = strax.SaveWhen.EXPLICIT

    dtype = [
        ("event_number", np.int64, "Event number in this dataset"),
        ("time", np.int64, "Event start time in ns since the unix epoch"),
        ("endtime", np.int64, "Event end time in ns since the unix epoch"),
    ]

    events_seen = 0

    electron_drift_velocity = straxen.URLConfig(
        default="cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    trigger_min_area = straxen.URLConfig(
        default=100,
        type=(int, float),
        help="Peaks must have more area (PE) than this to cause events",
    )

    trigger_max_competing = straxen.URLConfig(
        default=7,
        type=int,
        help="Peaks must have FEWER nearby larger or slightly smaller peaks to cause events",
    )

    left_event_extension = straxen.URLConfig(
        default=int(0.25e6),
        type=(int, float),
        help=(
            "Extend events this many ns to the left from each "
            "triggering peak. This extension is added to the maximum "
            "drift time."
        ),
    )

    right_event_extension = straxen.URLConfig(
        default=int(0.25e6),
        type=(int, float),
        help="Extend events this many ns to the right from each triggering peak.",
    )

    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z,
        type=(int, float),
        help="Total length of the TPC from the bottom of gate to the top of cathode wires [cm]",
    )

    exclude_s1_as_triggering_peaks = straxen.URLConfig(
        default=True,
        type=bool,
        help="If true exclude S1s as triggering peaks.",
    )

    event_s1_min_coincidence = straxen.URLConfig(
        default=2,
        infer_type=False,
        help=(
            "Event level S1 min coincidence. Should be >= "
            "s1_min_coincidence in the peaklet classification"
        ),
    )

    s1_min_coincidence = straxen.URLConfig(
        default=2, type=int, help="Minimum tight coincidence necessary to make an S1"
    )

    diagnose_overlapping = straxen.URLConfig(
        track=False, default=True, infer_type=False, help="Enable runtime checks for disjointness"
    )
    # shadow parameters
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

    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z,
        type=(int, float),
        help="Total length of the TPC from the bottom of gate to the top of cathode wires [cm]",
    )

    shadow_threshold = straxen.URLConfig(
        default={"s1_time_shadow": 1e3, "s2_time_shadow": 1e4, "s2_position_shadow": 1e4},
        type=dict,
        track=True,
        help="Only take S1/S2s larger than this into account when calculating Shadow [PE]",
    )

    electron_drift_velocity = straxen.URLConfig(
        default="cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    n_drift_time = straxen.URLConfig(
        default={"sr0": 1, "sr1": 2}, help="Number of drift time to veto"
    )

    sr = straxen.URLConfig(
        default="science_run://plugin.run_id?&phase=False",
        help="Science run to be used for the cut. It can affect the parameters of the cut.",
    )

    time_shadow_cut_values_mode = straxen.URLConfig(
        default={
            "bkg": [0.01010, 0.03822],  # Used by bkg and ted runs
            "ar37": [0.01010, 0.03822],
            "radon": [0.08303, 0.40418],
            "radon_hev": [0.33115, 0.61039],
            "ambe": [0.03078, 0.18748],
            "kr83m": [0.01479, 0.07908],
            "ybe": [0.03189, 0.09658],
            "rn222": [0.01178, 0.02782],
        },
        type=dict,
        track=True,
        help="cut values for 2 & 3 hits S1 in different run mode",
    )

    def setup(self):
        if self.s1_min_coincidence > self.event_s1_min_coincidence:
            raise ValueError(
                "Peak s1 coincidence requirement should be smaller "
                "or equal to event_s1_min_coincidence"
            )
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)
        # Left_extension and right_extension should be computed in setup to be
        # reflected in cutax too.
        self.left_extension = self.left_event_extension + self.drift_time_max
        self.right_extension = self.right_event_extension
        # shadow cuts parameters
        self.s2_area_limit = self.position_shadow_s2_area_limit
        self.ellipse_parameters = self.position_shadow_ellipse_parameters_mode[self.run_mode]
        self.straight_parameters = self.position_shadow_straight_parameters_mode[self.run_mode]
        self.veto_window = (
            self.max_drift_length / self.electron_drift_velocity * self.n_drift_time[self.sr]
        )
        self.time_shadow = self.time_shadow_cut_values_mode[self.run_mode]

    def get_window_size(self):
        # Take a large window for safety, events can have long tails
        return 10 * (self.left_event_extension + self.drift_time_max + self.right_event_extension)

    def _is_triggering(self, peaks):
        _is_triggering = peaks["area"] > self.trigger_min_area
        _is_triggering &= peaks["n_competing"] <= self.trigger_max_competing
        if self.exclude_s1_as_triggering_peaks:
            _is_triggering &= peaks["type"] == 2
        else:
            is_not_s1 = peaks["type"] != 1
            has_tc_large_enough = peaks["tight_coincidence"] >= self.event_s1_min_coincidence
            _is_triggering &= is_not_s1 | has_tc_large_enough
        # additionally require that the peak is tagged by the shadow cut etc.
        mask_good_exposure_peaks = self.get_good_exposure_mask(peaks)
        _is_triggering = _is_triggering & mask_good_exposure_peaks
        return _is_triggering

    def compute(self, peaks, start, end):
        _is_triggering = self._is_triggering(peaks)

        triggers = peaks[_is_triggering]

        # Join nearby triggers
        t0, t1 = strax.find_peak_groups(
            triggers,
            gap_threshold=self.left_extension + self.right_extension + 1,
            left_extension=self.left_extension,
            right_extension=self.right_extension,
        )

        # Don't extend beyond the chunk boundaries
        # This will often happen for events near the invalid boundary of the
        # overlap processing (which should be thrown away)
        t0 = np.clip(t0, start, end)
        t1 = np.clip(t1, start, end)

        result = np.zeros(len(t0), self.dtype)
        result["time"] = t0
        result["endtime"] = t1
        result["event_number"] = np.arange(len(result)) + self.events_seen

        if not result.size > 0:
            print("Found chunk without events?!")

        if self.diagnose_overlapping and len(result):
            # Check if the event windows overlap
            _event_window_do_not_overlap = (strax.endtime(result)[:-1] - result["time"][1:]) <= 0
            assert np.all(_event_window_do_not_overlap), "Events not disjoint"

        self.events_seen += len(result)

        return result

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
    def compute_position_shadow_cut(self, peaks):
        log10_time_shadow = np.log10(
            peaks["shadow_s2_position_shadow"] / peaks["pdf_s2_position_shadow"]
        )
        log10_pdf = np.log10(peaks["pdf_s2_position_shadow"])
        mask = ~(
            log10_time_shadow
            > self.prob_cut_line(log10_pdf, *self.ellipse_parameters, self.straight_parameters)
        )
        mask |= peaks["area"] > self.s2_area_limit
        mask |= peaks["type"] != 2
        return mask

    def compute_peak_time_veto(self, peaks):
        mask = np.full(len(peaks), True)
        for casting_peak in ["_s1", "_s2"]:
            mask &= (peaks[f"nearest_dt{casting_peak}"] > self.veto_window) | (
                peaks[f"nearest_dt{casting_peak}"] < 0
            )
        return mask

    def compute_peak_hotspot_veto(self, peaks):
        mask = peaks["se_score"] > 0.1
        mask &= peaks["type"] == 2
        return ~mask

    def compute_peak_time_shadow(self, peaks):
        # 2 hits
        time_shadow_2hits = peaks["shadow_s2_time_shadow"] > self.time_shadow[0]
        time_shadow_2hits &= peaks["n_hits"] == 2
        time_shadow_2hits &= peaks["type"] == 1
        # 3 hits
        time_shadow_3hits = peaks["shadow_s2_time_shadow"] > self.time_shadow[1]
        time_shadow_3hits &= peaks["n_hits"] >= 3
        time_shadow_3hits &= peaks["type"] == 1
        # for s2, cut on the small population
        s2_time_shadow_term = peaks["shadow_s2_position_shadow"] / peaks["pdf_s2_position_shadow"]
        s2_time_shadow_2hits = peaks["type"] == 2
        s2_time_shadow_2hits &= peaks["area"] < 300
        s2_time_shadow_2hits &= s2_time_shadow_term > self.time_shadow[0]
        return ~(time_shadow_2hits | time_shadow_3hits | s2_time_shadow_2hits)

    def get_good_exposure_mask(self, peaks):
        mask_good = self.compute_peak_hotspot_veto(peaks)
        mask_good &= self.compute_peak_time_veto(peaks)
        mask_good &= self.compute_peak_time_shadow(peaks)
        mask_good &= self.compute_position_shadow_cut(peaks)
        return mask_good
