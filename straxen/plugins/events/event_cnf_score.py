import numpy as np
import strax


class EventCNFScore(strax.Plugin):
    __version__ = "0.0.0"
    depends_on = ("event_basics", "peak_cnf_score")
    provides = "event_cnf_score"

    def infer_dtype(self):
        dtype = []
        for main_or_alt, description in zip(["", "alt_"], ["main", "alternate"]):
            dtype += [
                (
                    (
                        f"Maximum conditional normalizing flow score of {description} S2",
                        f"{main_or_alt}s2_cnf_score",
                    ),
                    np.float32,
                ),
                (
                    (
                        f"Time difference to the S2 casting maximum CNF score of {description} S2",
                        f"{main_or_alt}s2_cnf_nearest_dt",
                    ),
                    np.int64,
                ),
                (
                    (
                        f"S2 area of the S2 casting maximum CNF score of {description} S2",
                        f"{main_or_alt}s2_cnf_nearest_s2_area",
                    ),
                    np.float32,
                ),
                (
                    (
                        (
                            "Position difference to the S2 casting "
                            f"maximum CNF score of {description} S2"
                        ),
                        f"{main_or_alt}s2_cnf_nearest_dr",
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
            for type_ in ["s2", "alt_s2"]:
                type_index = event[f"{type_}_index"]
                if type_index != -1:
                    for p in [
                        "cnf_score",
                        "cnf_nearest_dt",
                        "cnf_nearest_s2_area",
                        "cnf_nearest_dr",
                    ]:
                        result[f"{type_}_{p}"][event_i] = sp[[p]][type_index]

        # 2. Set time and endtime for events
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        return result
