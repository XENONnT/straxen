import numpy as np
import strax

export, __all__ = strax.exporter()


@export
class EventNearestTriggering(strax.Plugin):
    """Time difference and properties of the nearest triggering peaks of main peaks of events."""

    __version__ = "0.0.0"
    depends_on = ("event_basics", "peak_basics", "peak_nearest_triggering")
    provides = "event_nearest_triggering"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = []
        common_descr = "of the nearest triggering peak on the"
        for main_peak, main_peak_desc in zip(["s1_", "s2_"], ["main S1", "main S2"]):
            for direction in ["left", "right"]:
                dtype += [
                    (
                        (
                            f"time difference {common_descr} {direction} of {main_peak_desc} [ns]",
                            f"{main_peak}{direction}_dtime",
                        ),
                        np.int64,
                    ),
                    (
                        (
                            f"time {common_descr} {direction} of {main_peak_desc} [ns]",
                            f"{main_peak}{direction}_time",
                        ),
                        np.int64,
                    ),
                    (
                        (
                            f"endtime {common_descr} {direction} of {main_peak_desc} [ns]",
                            f"{main_peak}{direction}_endtime",
                        ),
                        np.int64,
                    ),
                    (
                        (
                            f"center_time {common_descr} {direction} of {main_peak_desc} [ns]",
                            f"{main_peak}{direction}_center_time",
                        ),
                        np.int64,
                    ),
                    (
                        (
                            f"type {common_descr} {direction} of {main_peak_desc}",
                            f"{main_peak}{direction}_type",
                        ),
                        np.int8,
                    ),
                    (
                        (
                            f"proximity_score {common_descr} {direction} of {main_peak_desc}",
                            f"{main_peak}{direction}_proximity_score",
                        ),
                        np.float32,
                    ),
                    (
                        (
                            f"n_competing_left {common_descr} {direction} of {main_peak_desc}",
                            f"{main_peak}{direction}_n_competing_left",
                        ),
                        np.int32,
                    ),
                    (
                        (
                            f"n_competing {common_descr} {direction} of {main_peak_desc}",
                            f"{main_peak}{direction}_n_competing",
                        ),
                        np.int32,
                    ),
                    (
                        (
                            f"area {common_descr} {direction} of {main_peak_desc} [PE]",
                            f"{main_peak}{direction}_area",
                        ),
                        np.float32,
                    ),
                ]
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.dtype)

        strax.set_nan_defaults(result)

        # 1. Assign peaks features to main S1 and main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            res_i = result[event_i]
            # Fetch the features of main S1 and main S2
            for idx, main_peak in zip([event["s1_index"], event["s2_index"]], ["s1_", "s2_"]):
                if idx >= 0:
                    for direction in ["left", "right"]:
                        for field in [
                            "dtime",
                            "time",
                            "endtime",
                            "center_time",
                            "type",
                            "proximity_score",
                            "n_competing_left",
                            "n_competing",
                            "area",
                        ]:
                            res_i[f"{main_peak}{direction}_{field}"] = sp[f"{direction}_{field}"][
                                idx
                            ]

        # 2. Set time and endtime for events
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        return result
