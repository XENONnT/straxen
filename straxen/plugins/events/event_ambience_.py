import numpy as np
import strax


class EventAmbience_(strax.Plugin):
    __version__ = "0.0.0"
    depends_on = ("event_basics", "peak_ambience_")
    provides = "event_ambience_"

    @property
    def origin_dtype(self):
        return ["ambience_1d_score", "ambience_2d_score"]

    def infer_dtype(self):
        dtype = []
        for ambience, label in zip(self.origin_dtype, ["only in time", "in (time, space)"]):
            dtype.append(
                (
                    (f"Strength of ambient peaks {label} of main S1", f"s1_{ambience}"),
                    np.float32,
                )
            )
            dtype.append(
                (
                    (f"Strength of ambient peaks {label} of main S2", f"s2_{ambience}"),
                    np.float32,
                )
            )
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
                        result[f"{main_peak}{ambience}"][event_i] = sp[ambience][idx]

        # 3. Set time and endtime for events
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        return result
