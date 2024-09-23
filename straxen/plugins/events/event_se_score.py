import numpy as np
import strax


class EventSEScore(strax.Plugin):
    """This plugin is designed to calculate the single electron rate density for main and alt S2 in
    events.

    References:
        * v0.0.0: xenon:xenonnt:analysis:hot_spot_cut_summary
        * v0.1.0: xenon:xenonnt:ac:sr1:hotspot_veto_cut:wimp_roi

    """

    __version__ = "0.1.0"
    depends_on = ("event_basics", "peak_se_score")
    provides = "event_se_score"

    def infer_dtype(self):
        dtype = []
        for s_i in [1, 2]:
            for main_or_alt, description in zip(["", "alt_"], ["main", "alternate"]):
                dtype += [
                    (
                        (
                            f"Neiboring single-electron score of {description} S{s_i} ",
                            f"{main_or_alt}s{s_i}_se_score",
                        ),
                        np.float32,
                    ),
                ]
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.dtype)

        # 1. Assign peaks features to main S2 and alt S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            for type_ in ["s1", "alt_s1", "s2", "alt_s2"]:
                type_index = event[f"{type_}_index"]
                if type_index != -1:
                    result[f"{type_}_se_score"][event_i] = sp["se_score"][type_index]

        # 2. Set time and endtime for events
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        return result
