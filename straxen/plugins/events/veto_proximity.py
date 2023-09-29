import numpy as np
import straxen
import strax
import numba

export, __all__ = strax.exporter()


@export
class VetoProximity(strax.OverlapWindowPlugin):
    """Find the closest next/previous veto start w.r.t.

    the event time or when a busy happens during an event.
    """

    __version__ = "2.2.0"
    # Strictly speaking, we could depend on 'events', but then you couldn't
    # change the event_window_fields to e.g. s1_time and s2_endtime.
    depends_on = ("event_basics", "veto_intervals")
    provides = "veto_proximity"
    data_kind = "events"

    event_window_fields = straxen.URLConfig(
        default=("time", "endtime"),
        help="Fields to determine where to look for overlaps for using "
        "this plugin in the events. The default uses start and endtime "
        "of an event, but this can also be the S1 or S2 start/endtime",
    )

    veto_proximity_window = straxen.URLConfig(
        default=int(300e9), help="Maximum separation between veto stop and start pulses [ns]"
    )
    time_no_aqmon_veto_found = straxen.URLConfig(
        default=int(3.6e12),
        track=True,
        type=int,
        help="If no next/previous veto is found, we will fill the fields "
        "time_to_previous_XX with this time. Set to a large number "
        "such that one will never cut events that are < YY ns.",
    )

    veto_names = ["busy", "busy_he", "hev", "straxen_deadtime"]

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields
        start_field, stop_field = self.event_window_fields
        for name in self.veto_names:
            dtype += [
                (
                    (
                        f'Duration of event overlapping with "{name}"-veto [ns]',
                        f"veto_{name}_overlap",
                    ),
                    np.int64,
                ),
                (
                    (
                        f'Time (absolute value) to previous "{name}"-veto '
                        f'from "{start_field}" of event [ns]',
                        f"time_to_previous_{name}",
                    ),
                    np.int64,
                ),
                (
                    (
                        f'Time (absolute value) to next "{name}"-veto '
                        f'from "{stop_field}" of event [ns]',
                        f"time_to_next_{name}",
                    ),
                    np.int64,
                ),
            ]

        return dtype

    def get_window_size(self):
        return self.veto_proximity_window

    def set_result_for_veto(
        self,
        result_buffer: np.ndarray,
        event_window: np.ndarray,
        veto_intervals: np.ndarray,
        veto_name: str,
    ) -> None:
        """Fill the result buffer inplace. Goal is to find vetos with
        <veto_name> that are either during, before or after the current
        event_window.

        :param result_buffer: The buffer to fill inplace
        :param event_window: start/stop boundaries of the event to
            consider. Should be an array with ['time'] and ['endtime']
            which can be based on event start/end times or S1/S2 times
        :param veto_intervals: veto intervals datatype
        :param veto_name: The name of the veto to fill the result buffer
            for
        :return: Nothing, results are filled in place
        """
        # Set defaults to be some very long time
        result_buffer[f"time_to_previous_{veto_name}"] = self.time_no_aqmon_veto_found
        result_buffer[f"time_to_next_{veto_name}"] = self.time_no_aqmon_veto_found

        selected_intervals = veto_intervals[veto_intervals["veto_type"] == f"{veto_name}_veto"]
        if not len(selected_intervals):
            return

        vetos_during_event = strax.touching_windows(selected_intervals, event_window)

        # Figure out the vetos *during* an event
        res = self.get_overlapping_window_time(
            vetos_during_event, selected_intervals, event_window, result_buffer
        )
        result_buffer[f"veto_{veto_name}_overlap"] = res

        # Find the next and previous veto's
        times_to_prev, times_to_next = strax.abs_time_to_prev_next_interval(
            event_window, selected_intervals
        )
        mask_prev = times_to_prev > 0
        result_buffer[f"time_to_previous_{veto_name}"][mask_prev] = times_to_prev[mask_prev]

        max_next = times_to_next > 0
        result_buffer[f"time_to_next_{veto_name}"][max_next] = times_to_next[max_next]

    @staticmethod
    @numba.njit
    def get_overlapping_window_time(
        vetos_during_event, selected_intervals, event_window, result_buffer
    ):
        """Computes total time each event overlaps with the corresponding
        veto."""
        res = np.zeros(len(vetos_during_event), np.int64)

        for event_i, veto_window in enumerate(vetos_during_event):
            if veto_window[1] - veto_window[0]:
                vetos_in_window = selected_intervals[veto_window[0] : veto_window[1]].copy()
                starts = np.clip(
                    vetos_in_window["time"],
                    event_window[event_i]["time"],
                    event_window[event_i]["endtime"],
                )
                stops = np.clip(
                    vetos_in_window["endtime"],
                    event_window[event_i]["time"],
                    event_window[event_i]["endtime"],
                )

                res[event_i] = np.sum(stops - starts)
        return res

    def compute(self, events, veto_intervals):
        result = np.zeros(len(events), self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        # Get containers for touching windows based on self.event_window_fields
        event_window = np.zeros(len(events), dtype=strax.time_fields)
        event_window["time"] = events[self.event_window_fields[0]]
        event_window["endtime"] = events[self.event_window_fields[1]]

        for veto_name in self.veto_names:
            self.set_result_for_veto(result, event_window, veto_intervals, veto_name)
        return result
