import numpy as np
import strax

export, __all__ = strax.exporter()


@export
class DistinctChannels(strax.LoopPlugin):
    """Compute the number of contributing PMTs that contribute to the alt_s1 but not to the main
    S1."""

    __version__ = "0.1.1"
    depends_on = ("event_basics", "peaks")
    loop_over = "events"
    dtype = [
        (
            "alt_s1_distinct_channels",
            np.int32,
            "Number of PMTs contributing to the secondary S1 "
            "that do not contribute to the main S1",
        ),
    ] + strax.time_fields

    def compute_loop(self, event, peaks):
        if event["alt_s1_index"] == -1:
            n_distinct = 0
        else:
            s1_a = peaks[event["s1_index"]]
            s1_b = peaks[event["alt_s1_index"]]
            s1_a_peaks = np.nonzero((s1_a["area_per_channel"] > 0) * 1)
            s1_b_peaks = np.nonzero((s1_b["area_per_channel"] > 0) * 1)
            n_distinct = 0
            for channel in range(len(s1_b_peaks[0])):
                if s1_b_peaks[0][channel] not in s1_a_peaks[0]:
                    n_distinct += 1

        return dict(
            alt_s1_distinct_channels=n_distinct, time=event["time"], endtime=event["endtime"]
        )
