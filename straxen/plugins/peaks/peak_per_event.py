import numpy as np
import strax
import straxen


export, __all__ = strax.exporter()


@export
class EventPeaks(strax.Plugin):
    """Add event number for peaks and drift times of all s2 depending on the largest s1.

    Link - https://xe1t-wiki.lngs.infn.it/doku.php?id=weiss:analysis:ms_plugin

    """

    __version__ = "0.0.1"
    depends_on = ("event_basics", "peak_basics", "peak_positions")
    provides = "peak_per_event"
    data_kind = "peaks"
    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        dtype = strax.time_fields + [
            ("drift_time", np.float32, "Drift time between main S1 and S2 in ns"),
            ("event_number", np.int64, "Event number in this dataset"),
        ]
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        split_peaks_ind = strax.fully_contained_in(peaks, events)
        result = np.zeros(len(peaks), self.dtype)
        straxen.EventBasics.set_nan_defaults(result)

        # Assign peaks features to main S1 and main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            result["drift_time"][split_peaks_ind == event_i] = (
                sp["center_time"] - event["s1_center_time"]
            )
        result["event_number"] = split_peaks_ind
        result["drift_time"][peaks["type"] != 2] = np.nan
        result["time"] = peaks["time"]
        result["endtime"] = strax.endtime(peaks)
        return result
