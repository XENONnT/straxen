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
        2. The triggering peak must have less than "trigger_max_proximity".

    Note:
        The time range which defines an event gets chopped at the chunk
        boundaries. This happens at invalid boundaries of the

    """

    __version__ = "0.1.1"

    depends_on = ("peak_basics", "peak_proximity")
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
        default="xedocs://electron_drift_velocities?attr=value&run_id=plugin.run_id&version=ONLINE",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    trigger_min_area = straxen.URLConfig(
        default=120,
        type=(int, float),
        help="Peaks must have more area (PE) than this to cause events",
    )

    trigger_feature = straxen.URLConfig(
        default="proximity_score",
        type=str,
        help="Feature as event building criterion",
    )

    trigger_max_proximity = straxen.URLConfig(
        default=1.5e-3,
        type=(int, float),
        help="Peaks must have less proximity score to cause events",
    )

    # TODO: extand the padding to cover all multiple scattering
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

    event_exclude_s3 = straxen.URLConfig(
        default=True,
        type=bool,
        help="Event will not be built if S3 is present in the window",
    )

    diagnose_overlapping = straxen.URLConfig(
        track=False, default=True, infer_type=False, help="Enable runtime checks for disjointness"
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

    def get_window_size(self):
        # Take a large window for safety, events can have long tails
        return 10 * (self.left_event_extension + self.drift_time_max + self.right_event_extension)

    def _is_triggering(self, peaks):
        _is_triggering = peaks["area"] > self.trigger_min_area
        _is_triggering &= peaks["type"] == 2
        _is_triggering &= peaks[self.trigger_feature] <= self.trigger_max_proximity
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
        # The different chunking causes different results, but this is
        # NOT tracked by lineage
        t0 = np.clip(t0, start, end)
        t1 = np.clip(t1, start, end)

        result = np.zeros(len(t0), self.dtype)
        result["time"] = t0
        result["endtime"] = t1

        if self.event_exclude_s3:
            # type 3 high energy gas scintillation xenon:xenonnt:lsanchez:som_sr1b_summary_note
            split_peaks = strax.split_by_containment(peaks, result)
            has_s3 = np.array([np.any(sp["type"] == 3) for sp in split_peaks]).astype(bool)
            result = result[~has_s3]

        # because this is an overlap window plugin, event_number is not continuous
        result["event_number"] = np.arange(len(result)) + self.events_seen
        self.events_seen += len(result)

        if not result.size > 0:
            print("Found chunk without events?!")

        if self.diagnose_overlapping and len(result):
            # Check if the event windows overlap
            _event_window_do_not_overlap = (strax.endtime(result)[:-1] - result["time"][1:]) <= 0
            assert np.all(_event_window_do_not_overlap), "Events not disjoint"

        return result
