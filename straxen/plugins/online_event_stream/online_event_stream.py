import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@export
class OnlineEventStream(strax.Plugin):
    """Compact event stream for website/monitoring use.

    This plugin is designed for Mongo online storage and keeps only a small,
    deterministic subset of fields from clean events in each chunk.
    """

    __version__ = "0.0.1"

    depends_on = ("event_basics", "event_area_per_channel", "event_waveform")
    provides = "online_event_stream"
    data_kind = "online_event_stream"

    online_event_stream_selection = straxen.URLConfig(
        type=str,
        default=(
            "(s1_index >= 0) & (s2_index >= 0) & (drift_time > 0)"
            " & (s1_area > 30) & (s2_area > 500)"
        ),
        help="Selection string for events included in the online event stream.",
    )

    online_event_stream_max_events_per_chunk = straxen.URLConfig(
        type=int,
        default=200,
        help=(
            "Maximum number of selected events stored per chunk. "
            "Use a non-positive value to keep all selected events."
        ),
    )

    online_event_stream_top_n_channels = straxen.URLConfig(
        type=int,
        default=16,
        help="Number of top channels kept for S1/S2 hit-pattern summaries.",
    )

    online_event_stream_max_bytes = straxen.URLConfig(
        type=int,
        default=8_000_000,
        help="Hard cap on bytes stored per chunk for this online data product.",
    )

    def infer_dtype(self):
        n_top = self.online_event_stream_top_n_channels
        dtype = strax.time_fields + [
            (("Event number", "event_number"), np.int64),
            (("S1 area [PE]", "s1_area"), np.float32),
            (("S2 area [PE]", "s2_area"), np.float32),
            (("S1 area fraction top", "s1_area_fraction_top"), np.float32),
            (("S2 area fraction top", "s2_area_fraction_top"), np.float32),
            (("S2 reconstructed x [cm]", "s2_x"), np.float32),
            (("S2 reconstructed y [cm]", "s2_y"), np.float32),
            (("Drift time [ns]", "drift_time"), np.float32),
            (("S1 contributing channels", "s1_n_channels"), np.int16),
            (("S2 contributing channels", "s2_n_channels"), np.int16),
            (
                ("S1 top channel indices by area contribution", "s1_top_channel_index"),
                np.int16,
                n_top,
            ),
            (("S1 top channel areas [PE]", "s1_top_channel_area"), np.float32, n_top),
            (
                ("S2 top channel indices by area contribution", "s2_top_channel_index"),
                np.int16,
                n_top,
            ),
            (("S2 top channel areas [PE]", "s2_top_channel_area"), np.float32, n_top),
            (
                ("Fraction of selected events retained in this chunk", "stored_fraction"),
                np.float32,
            ),
        ]
        wf_dtype = self.deps["event_waveform"].dtype_for("event_waveform")
        dtype += [
            (("Main S1 waveform [PE/sample]", "s1_data"), wf_dtype.fields["s1_data"][0]),
            (("Main S2 waveform [PE/sample]", "s2_data"), wf_dtype.fields["s2_data"][0]),
            (("Alt S1 waveform [PE/sample]", "alt_s1_data"), wf_dtype.fields["alt_s1_data"][0]),
            (("Alt S2 waveform [PE/sample]", "alt_s2_data"), wf_dtype.fields["alt_s2_data"][0]),
        ]
        return dtype

    def compute(self, event_basics, event_area_per_channel, event_waveform):
        if len(event_basics) != len(event_area_per_channel) or len(event_basics) != len(event_waveform):
            raise ValueError(
                "event_basics, event_area_per_channel and event_waveform must have the same length, "
                f"got {len(event_basics)}, {len(event_area_per_channel)}, {len(event_waveform)}"
            )
        if len(event_basics) and not np.array_equal(
            event_basics["time"], event_area_per_channel["time"]
        ):
            raise ValueError("event_basics and event_area_per_channel are not aligned in time.")
        if len(event_basics) and not np.array_equal(event_basics["time"], event_waveform["time"]):
            raise ValueError("event_basics and event_waveform are not aligned in time.")

        if not len(event_basics):
            return np.zeros(0, dtype=self.dtype)

        selection = self.online_event_stream_selection
        if selection:
            mask = strax.parse_selection(event_basics, selection)
        else:
            mask = np.ones(len(event_basics), dtype=np.bool_)

        selected = np.flatnonzero(mask)
        if not len(selected):
            return np.zeros(0, dtype=self.dtype)

        stored_fraction = np.float32(1.0)
        max_events = self.online_event_stream_max_events_per_chunk
        if max_events > 0 and len(selected) > max_events:
            stored_fraction = np.float32(max_events / len(selected))
            selected = selected[:max_events]

        # Extra safety for Mongo document size.
        bytes_per_row = np.dtype(self.dtype).itemsize
        max_by_bytes = self.online_event_stream_max_bytes // bytes_per_row
        if max_by_bytes <= 0:
            return np.zeros(0, dtype=self.dtype)
        if len(selected) > max_by_bytes:
            stored_fraction *= np.float32(max_by_bytes / len(selected))
            selected = selected[:max_by_bytes]

        result = np.zeros(len(selected), dtype=self.dtype)
        strax.set_nan_defaults(result)

        selected_events = event_basics[selected]
        selected_apc = event_area_per_channel[selected]
        selected_wf = event_waveform[selected]

        result["time"] = selected_events["time"]
        result["endtime"] = strax.endtime(selected_events)
        result["event_number"] = selected_events["event_number"]
        result["s1_area"] = selected_events["s1_area"]
        result["s2_area"] = selected_events["s2_area"]
        result["s1_area_fraction_top"] = selected_events["s1_area_fraction_top"]
        result["s2_area_fraction_top"] = selected_events["s2_area_fraction_top"]
        result["s2_x"] = selected_events["s2_x"]
        result["s2_y"] = selected_events["s2_y"]
        result["drift_time"] = selected_events["drift_time"]
        result["s1_n_channels"] = selected_events["s1_n_channels"]
        result["s2_n_channels"] = selected_events["s2_n_channels"]
        result["stored_fraction"] = stored_fraction
        result["s1_data"] = selected_wf["s1_data"]
        result["s2_data"] = selected_wf["s2_data"]
        result["alt_s1_data"] = selected_wf["alt_s1_data"]
        result["alt_s2_data"] = selected_wf["alt_s2_data"]

        for i, row in enumerate(selected_apc):
            s1_idx, s1_area = self._top_channels(row["s1_area_per_channel"])
            s2_idx, s2_area = self._top_channels(row["s2_area_per_channel"])
            result["s1_top_channel_index"][i] = s1_idx
            result["s1_top_channel_area"][i] = s1_area
            result["s2_top_channel_index"][i] = s2_idx
            result["s2_top_channel_area"][i] = s2_area

        return result

    def _top_channels(self, area_per_channel):
        n_top = self.online_event_stream_top_n_channels
        idx = np.full(n_top, -1, dtype=np.int16)
        values = np.zeros(n_top, dtype=np.float32)

        if not len(area_per_channel):
            return idx, values

        order = np.argsort(area_per_channel)[::-1]
        order = order[:n_top]
        vals = area_per_channel[order]

        valid = vals > 0
        order = order[valid]
        vals = vals[valid]

        idx[: len(order)] = order.astype(np.int16, copy=False)
        values[: len(vals)] = vals.astype(np.float32, copy=False)
        return idx, values
