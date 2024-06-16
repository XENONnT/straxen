import strax
import straxen
import numpy as np


MAX_NUMBER_OF_PEAKS_PER_EVENT = 10
PEAK_WAVEFORM_LENGTH = 200


class MultiPeakData(strax.Plugin):
    """Plugin that extracts information of multiple peaks in an event."""

    __version__ = "0.0.1"

    depends_on = ("peaks", "peak_per_event", "events", "peak_positions", "peak_basics")
    provides = "multi_peak_data"
    data_kind = "events"
    save_when = strax.SaveWhen.TARGET

    dtype = [
        (
            ("time difference to the first peak in the event in ns", "delta_time_i"),
            np.int64,
            MAX_NUMBER_OF_PEAKS_PER_EVENT,
        ),
        (
            ("sample width in ns of the corresponding waveform", "peak_dt_i"),
            np.int32,
            MAX_NUMBER_OF_PEAKS_PER_EVENT,
        ),
        (
            ("peak type of the corresponding waveform", "peak_type_i"),
            np.int8,
            MAX_NUMBER_OF_PEAKS_PER_EVENT,
        ),
        (
            ("peak area of the corresponding waveform", "peak_area_i"),
            np.float32,
            MAX_NUMBER_OF_PEAKS_PER_EVENT,
        ),
        (("x position of peak i", "peak_x_position_i"), np.float32, MAX_NUMBER_OF_PEAKS_PER_EVENT),
        (("y position of peak i", "peak_y_position_i"), np.float32, MAX_NUMBER_OF_PEAKS_PER_EVENT),
        (("area fraction top of peak i", "peak_aft_i"), np.float32, MAX_NUMBER_OF_PEAKS_PER_EVENT),
        (
            ("Sum Waveforms of all peaks in the event", "peak_waveform_i"),
            np.int16,
            (MAX_NUMBER_OF_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH),
        ),
        (
            ("Top Waveforms of all peaks in the event", "peak_waveform_i_top"),
            np.int16,
            (MAX_NUMBER_OF_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH),
        ),
        (
            ("PMT hitpattern of all peaks in the event", "peak_area_per_channel_i"),
            np.float32,
            (MAX_NUMBER_OF_PEAKS_PER_EVENT, straxen.n_tpc_pmts),
        ),
        *strax.time_fields,
    ]

    def compute(self, peaks, events):

        # Just interested in S1 or S2 peaks. Here would be the place to apply further peak quality cuts.
        peaks = peaks[peaks["type"] != 0]

        peaks_per_event = strax.split_by_containment(peaks, events)

        result = np.zeros(len(events), dtype=self.dtype)

        for i, peaks_in_event in enumerate(peaks_per_event):

            n_peaks_in_event = len(peaks_in_event)

            if n_peaks_in_event == 0:
                continue

            else:
                result[i]["delta_time_i"][0:n_peaks_in_event] = (
                    peaks_in_event["time"][0:MAX_NUMBER_OF_PEAKS_PER_EVENT]
                    - peaks_in_event["time"][0]
                )
                result[i]["peak_dt_i"][0:n_peaks_in_event] = peaks_in_event["dt"][
                    0:MAX_NUMBER_OF_PEAKS_PER_EVENT
                ]
                result[i]["peak_area_per_channel_i"][0:n_peaks_in_event] = peaks_in_event[
                    "area_per_channel"
                ][0:MAX_NUMBER_OF_PEAKS_PER_EVENT]
                result[i]["peak_waveform_i"][0:n_peaks_in_event] = peaks_in_event["data"][
                    0:MAX_NUMBER_OF_PEAKS_PER_EVENT
                ]
                result[i]["peak_waveform_i_top"][0:n_peaks_in_event] = peaks_in_event["data_top"][
                    0:MAX_NUMBER_OF_PEAKS_PER_EVENT
                ]
                result[i]["peak_type_i"][0:n_peaks_in_event] = peaks_in_event["type"][
                    0:MAX_NUMBER_OF_PEAKS_PER_EVENT
                ]
                result[i]["peak_area_i"][0:n_peaks_in_event] = peaks_in_event["area"][
                    0:MAX_NUMBER_OF_PEAKS_PER_EVENT
                ]
                result[i]["peak_x_position_i"][0:n_peaks_in_event] = peaks_in_event["x"][
                    0:MAX_NUMBER_OF_PEAKS_PER_EVENT
                ]
                result[i]["peak_y_position_i"][0:n_peaks_in_event] = peaks_in_event["y"][
                    0:MAX_NUMBER_OF_PEAKS_PER_EVENT
                ]
                result[i]["peak_aft_i"][0:n_peaks_in_event] = peaks_in_event["area_fraction_top"][
                    0:MAX_NUMBER_OF_PEAKS_PER_EVENT
                ]

        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        return result
