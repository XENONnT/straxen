import strax
import numpy as np

MAX_NUMBER_OF_S1_PEAKS_PER_EVENT = 5
MAX_NUMBER_OF_S2_PEAKS_PER_EVENT = 5
PEAK_WAVEFORM_LENGTH = 200


class MultiPeakData(strax.Plugin):
    """Plugin that extracts information of multiple peaks in an event."""

    __version__ = "0.0.2"

    depends_on = ("peaks", "peak_per_event", "events", "peak_positions", "peak_basics")
    provides = "multi_peak_data"
    data_kind = "events"
    save_when = strax.SaveWhen.TARGET

    dtype = [
        (
            ("time difference of S1 peaks to event start time", "s1_delta_time_i"),
            np.int64,
            MAX_NUMBER_OF_S1_PEAKS_PER_EVENT,
        ),
        (
            ("time difference of S2 peaks to event start time", "s2_delta_time_i"),
            np.int64,
            MAX_NUMBER_OF_S2_PEAKS_PER_EVENT,
        ),
        (
            ("sample width in ns of the S1 waveform", "s1_peak_dt_i"),
            np.int32,
            MAX_NUMBER_OF_S1_PEAKS_PER_EVENT,
        ),
        (
            ("sample width in ns of the S2 waveform", "s2_peak_dt_i"),
            np.int32,
            MAX_NUMBER_OF_S2_PEAKS_PER_EVENT,
        ),
        (("x position of S2 i", "s2_x_position_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("y position of S2 i", "s2_y_position_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("area fraction top of S2 i", "s2_aft_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("area fraction top of S1 i", "s1_aft_i"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (
            ("Sum Waveform of S1 peaks", "s1_waveform_i"),
            np.float32,
            (MAX_NUMBER_OF_S1_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH),
        ),
        (
            ("Sum Waveform of S2 peaks", "s2_waveform_i"),
            np.float32,
            (MAX_NUMBER_OF_S2_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH),
        ),
        (
            ("Start of S1 waverforms", "s1_waveform_start_i"),
            np.float32,
            (MAX_NUMBER_OF_S1_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH),
        ),
        *strax.time_fields,
    ]

    def compute(self, peaks, events):

        # Just interested in S1 or S2 peaks. 
        # Here would be the place to apply further peak quality cuts.
        peaks = peaks[peaks["type"] != 0]

        peaks_per_event = strax.split_by_containment(peaks, events)

        result = np.zeros(len(events), dtype=self.dtype)

        for i, peaks_in_event in enumerate(peaks_per_event):

            n_peaks_in_event = len(peaks_in_event)

            if n_peaks_in_event == 0:
                continue
            else:

                # Get the largest S1s and S2s
                s1_peaks = peaks_in_event[peaks_in_event["type"] == 1]
                s2_peaks = peaks_in_event[peaks_in_event["type"] == 2]

                s1_peaks = s1_peaks[np.argsort(s1_peaks["area"])[::-1]]
                s2_peaks = s2_peaks[np.argsort(s2_peaks["area"])[::-1]]

                n_s1_peaks_in_event = len(s1_peaks)
                n_s2_peaks_in_event = len(s2_peaks)

                # Save the data of the n largest peaks
                result[i]["s1_delta_time_i"][0:n_s1_peaks_in_event] = (
                    s1_peaks["time"][0:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT] - peaks_in_event["time"][0]
                )
                result[i]["s2_delta_time_i"][0:n_s2_peaks_in_event] = (
                    s2_peaks["time"][0:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT] - peaks_in_event["time"][0]
                )

                result[i]["s1_peak_dt_i"][0:n_s1_peaks_in_event] = s1_peaks["dt"][
                    0:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT
                ]
                result[i]["s2_peak_dt_i"][0:n_s2_peaks_in_event] = s2_peaks["dt"][
                    0:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT
                ]

                result[i]["s2_x_position_i"][0:n_s2_peaks_in_event] = s2_peaks["x"][
                    0:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT
                ]
                result[i]["s2_y_position_i"][0:n_s2_peaks_in_event] = s2_peaks["y"][
                    0:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT
                ]
                result[i]["s2_aft_i"][0:n_s2_peaks_in_event] = s2_peaks["area_fraction_top"][
                    0:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT
                ]

                result[i]["s1_aft_i"][0:n_s1_peaks_in_event] = s1_peaks["area_fraction_top"][
                    0:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT
                ]

                result[i]["s1_waveform_i"][0:n_s1_peaks_in_event] = s1_peaks["data"][
                    0:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT
                ]
                result[i]["s2_waveform_i"][0:n_s2_peaks_in_event] = s2_peaks["data"][
                    0:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT
                ]

                result[i]["s1_waveform_start_i"][0:n_s1_peaks_in_event] = s1_peaks["data_start"][
                    0:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT
                ]

        # If waveform_start was not explicitly saved, we can take it from the regular waveform
        mask = result["s1_peak_dt_i"] == 10
        result["s1_waveform_start_i"][mask] = result["s1_waveform_i"][mask]

        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        return result
