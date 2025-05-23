import numpy as np
import strax


class EventAmbience(strax.Plugin):
    """Save Ambience of the main S1 and main S2 in the event.

    References:
        * v0.0.4 reference: xenon:xenonnt:ac:prediction:shadow_ambience
        * v0.1.0 reference: xenon:xenonnt:analysis:redefine_n_competing

    """

    __version__ = "0.1.0"
    depends_on = ("event_basics", "peak_ambience")
    provides = "event_ambience"

    @property
    def origin_dtype(self):
        return ["lh_before", "s0_before", "s1_before", "s2_before", "s2_near"]

    def infer_dtype(self):
        dtype = []
        for ambience in self.origin_dtype:
            dtype.append(
                (
                    (f"Number of {' '.join(ambience.split('_'))} main S1", f"s1_n_{ambience}"),
                    np.int16,
                )
            )
            dtype.append(
                (
                    (f"Number of {' '.join(ambience.split('_'))} main S2", f"s2_n_{ambience}"),
                    np.int16,
                )
            )
        dtype += [
            (
                ("Sum of small hits and peaks before main S1", "s1_s_before"),
                np.float32,
            ),
            (
                ("Sum of small hits and peaks before main S2", "s2_s_before"),
                np.float32,
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)

        # 1. Initialization, ambience is set to be the lowest possible value
        result = np.zeros(len(events), self.dtype)
        # 2. Assign peaks features to main S1, main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            for idx, main_peak in zip([event["s1_index"], event["s2_index"]], ["s1_", "s2_"]):
                if idx >= 0:
                    for ambience in self.origin_dtype:
                        result[f"{main_peak}n_{ambience}"][event_i] = sp[f"n_{ambience}"][idx]
                    result[f"{main_peak}s_before"][event_i] = sp["s_before"][idx]

        # 3. Set time and endtime for events
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        return result
