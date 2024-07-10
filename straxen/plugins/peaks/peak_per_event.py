import strax
import numpy as np

import straxen

export, __all__ = strax.exporter()


@export
class EventPeaks(strax.Plugin):
    """
    0.0.2 , Add event number for peaks and drift times of all s2 depending on the largest s1.
    Link - https://xe1t-wiki.lngs.infn.it/doku.php?id=weiss:analysis:ms_plugin
    """

    __version__ = "0.0.2"
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
        """
        Start changed part
        This section of the code ensures consistency between event numbers in peaks and events.

        We start by sorting the indices of fully contained peaks. Then, we create a dictionary called 'mapping',
        where keys represent event numbers in peak level data, and values represent event numbers in event level data.

        The 'mapping' dictionary ensures that each event number at the peak level corresponds to the correct event
        number at the event level. We then use 'corrected_split_peaks_ind' to update the event numbers of
        peaks based on the 'mapping' dictionary. This process ensures that event numbers match between peaks and events.

        The 'result['event_number']' array is updated with 'corrected_split_peaks_ind', ensuring that event numbers
        in peaks are consistent with those in events, maintaining the original length and preserving -1 values for peaks
        where event numbers are not applicable.
        """
        sorted_indices_split_peaks_ind = np.argsort(
            split_peaks_ind
        )  # Sort the indices of fully contained peaks for mapping
        mapping = {
            val: events["event_number"][i]
            for val,  # Create a mapping dictionary to match event numbers in peaks
            ##to event numbers in events
            i in zip(
                np.unique(
                    split_peaks_ind[sorted_indices_split_peaks_ind][
                        split_peaks_ind[sorted_indices_split_peaks_ind] != -1
                    ]
                ),
                range(len(events["event_number"])),
            )
        }
        # Correct event numbers of peaks based on the mapping dictionary
        corrected_split_peaks_ind = np.array(
            [mapping[val] if val in mapping else val for val in split_peaks_ind]
        )  ##  we run on all numbers in split_paeaks_ind and call them val
        # Update the event_number field in the result array with corrected event numbers for peaks
        result["event_number"] = corrected_split_peaks_ind
        """End changed part."""
        result["drift_time"][peaks["type"] != 2] = np.nan
        result["time"] = peaks["time"]
        result["endtime"] = strax.endtime(peaks)
        return result
