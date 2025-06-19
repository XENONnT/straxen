import strax
import numpy as np


class EventwBayesClass(strax.Plugin):
    """Append at event level the posterior probability for an S1, S2, alt_S1 and alt_S2."""

    provides = "event_w_bayes_class"
    depends_on = ("peak_classification_bayes", "event_basics")
    data_kind = "events"
    __version__ = "0.0.1"

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields
        for name in ["s1", "s2", "alt_s1", "alt_s2"]:
            dtype += [(f"{name}_ln_prob_s1", np.float32, f"Given an {name}, s1 ln probability")]
            dtype += [(f"{name}_ln_prob_s2", np.float32, f"Given an {name}, s2 ln probability")]

        return dtype

    def compute(self, peaks, events):
        split_peaks = strax.split_by_containment(peaks, events)

        # 1. Initialization
        result = np.empty(len(events), dtype=self.dtype)
        # 2. Set time and endtime for events
        for name in ["s1", "s2", "alt_s1", "alt_s2"]:
            result[f"{name}_ln_prob_s1"] = np.nan
            result[f"{name}_ln_prob_s2"] = np.nan
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]
        # 3. Assign peaks features to main S1, main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            for name in ["s1", "s2", "alt_s1", "alt_s2"]:
                if event[f"{name}_index"] >= 0:
                    result[f"{name}_ln_prob_s1"][event_i] = sp["ln_prob_s1"][event[f"{name}_index"]]
                    result[f"{name}_ln_prob_s2"][event_i] = sp["ln_prob_s2"][event[f"{name}_index"]]
        return result
