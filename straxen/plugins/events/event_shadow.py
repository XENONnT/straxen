import numpy as np
import strax

export, __all__ = strax.exporter()


@export
class EventShadow(strax.Plugin):
    """This plugin can calculate shadow for main S1 and main S2 in events. It also gives the
    position information of the previous peaks.

    References:
        * v0.1.4 reference: xenon:xenonnt:ac:prediction:shadow_ambience

    """

    __version__ = "0.1.4"
    depends_on = ("event_basics", "peak_shadow")
    provides = "event_shadow"

    def infer_dtype(self):
        dtype = []
        for main_peak, main_peak_desc in zip(["s1_", "s2_"], ["main S1", "main S2"]):
            # previous S1 can only cast time shadow,
            # previous S2 can cast both time & position shadow
            for key in ["s1_time_shadow", "s2_time_shadow", "s2_position_shadow"]:
                type_str, tp_desc, _ = key.split("_")
                dtype.append(
                    (
                        (
                            (
                                f"Largest {tp_desc} shadow casting from previous {type_str} to"
                                f" {main_peak_desc} [PE/ns]"
                            ),
                            f"{main_peak}shadow_{key}",
                        ),
                        np.float32,
                    )
                )
                dtype.append(
                    (
                        (
                            (
                                f"Time difference from the previous {type_str} casting largest"
                                f" {tp_desc} shadow to {main_peak_desc} [ns]"
                            ),
                            f"{main_peak}dt_{key}",
                        ),
                        np.int64,
                    )
                )
                # Only previous S2 peaks have (x,y)
                if "s2" in key:
                    dtype.append(
                        (
                            (
                                (
                                    f"X of previous s2 peak casting largest {tp_desc} shadow on"
                                    f" {main_peak_desc} [cm]"
                                ),
                                f"{main_peak}x_{key}",
                            ),
                            np.float32,
                        )
                    )
                    dtype.append(
                        (
                            (
                                (
                                    f"Y of previous s2 peak casting largest {tp_desc} shadow on"
                                    f" {main_peak_desc} [cm]"
                                ),
                                f"{main_peak}y_{key}",
                            ),
                            np.float32,
                        )
                    )
                # Only time shadow gives the nearest large peak
                if "time" in key:
                    dtype.append(
                        (
                            (
                                (f"The nearest previous large {type_str} to {main_peak_desc} [ns]"),
                                f"{main_peak}nearest_{type_str}",
                            ),
                            np.float32,
                        )
                    )
                    dtype.append(
                        (
                            (
                                (
                                    f"Time difference from the nearest previous large {type_str} to"
                                    f" {main_peak_desc} [ns]"
                                ),
                                f"{main_peak}nearest_dt_{type_str}",
                            ),
                            np.int64,
                        )
                    )
            # Also record the PDF of HalfCauchy when calculating S2 position shadow
            dtype.append(
                (
                    (
                        f"PDF describing correlation between previous s2 and {main_peak_desc}",
                        f"{main_peak}pdf_s2_position_shadow",
                    ),
                    np.float32,
                )
            )
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
                    for key in ["s1_time_shadow", "s2_time_shadow", "s2_position_shadow"]:
                        type_str = key.split("_")[0]
                        res_i[f"{main_peak}shadow_{key}"] = sp[f"shadow_{key}"][idx]
                        res_i[f"{main_peak}dt_{key}"] = sp[f"dt_{key}"][idx]
                        if "time" in key:
                            res_i[f"{main_peak}nearest_{type_str}"] = sp[f"nearest_{type_str}"][idx]
                            res_i[f"{main_peak}nearest_dt_{type_str}"] = sp[
                                f"nearest_dt_{type_str}"
                            ][idx]
                        if "s2" in key:
                            res_i[f"{main_peak}x_{key}"] = sp[f"x_{key}"][idx]
                            res_i[f"{main_peak}y_{key}"] = sp[f"y_{key}"][idx]
                    # Record the PDF of HalfCauchy
                    res_i[f"{main_peak}pdf_s2_position_shadow"] = sp["pdf_s2_position_shadow"][idx]

        # 2. Set time and endtime for events
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        return result
