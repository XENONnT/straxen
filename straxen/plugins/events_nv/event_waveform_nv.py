import strax
import straxen
import numpy as np
from immutabledict import immutabledict

export, __all__ = strax.exporter()


@export
class nVETOEventWaveform(strax.Plugin):
    """Plugin which computes the summed waveform as well as some shape properties of the NV
    events."""

    __version__ = "0.0.1"

    depends_on = "events_nv", "records_nv"
    provides = "event_waveform_nv"
    data_kind = "events_nv"
    compressor = "zstd"

    gain_model_nv = straxen.URLConfig(
        default="cmt://to_pe_model_nv?version=ONLINE&run_id=plugin.run_id",
        infer_type=False,
        help="PMT gain model. Specify as (model_type, model_config, nT = True)",
    )

    channel_map = straxen.URLConfig(
        track=False,
        type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) channel number",
    )

    def infer_dtype(self):
        return veto_event_waveform_dtype()

    def setup(self):
        self.channel_range = self.channel_map["nveto"]
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1

        to_pe = self.gain_model_nv

        # Create to_pe array of size max channel:
        self.to_pe = np.zeros(self.channel_range[1] + 1, dtype=np.float32)
        self.to_pe[self.channel_range[0] :] = to_pe[:]

    def compute(self, events_nv, records_nv, start, end):
        events_waveform = np.zeros(len(events_nv), self.dtype)

        # Compute shape like properties:
        _tmp_events = np.zeros(len(events_nv), dtype=_temp_event_data_type())
        strax.copy_to_buffer(events_nv, _tmp_events, "_temp_nv_evts_wf_cpy")
        _tmp_events["length"] = (events_nv["endtime"] - events_nv["time"]) // 2
        _tmp_events["dt"] = 2
        strax.simple_summed_waveform(records_nv, _tmp_events, self.to_pe)
        strax.compute_properties(_tmp_events)

        strax.copy_to_buffer(_tmp_events, events_waveform, "_temp_nv_evts_cpy")
        events_waveform["range_50p_area"] = _tmp_events["width"][:, 5]
        events_waveform["range_90p_area"] = _tmp_events["width"][:, 9]
        events_waveform["rise_time"] = -_tmp_events["area_decile_from_midpoint"][:, 1]
        del _tmp_events

        return events_waveform


def veto_event_waveform_dtype(
    n_samples_wf: int = 200,
) -> list:
    dtype = strax.time_dt_fields + [
        (("Waveform data in PE/sample (not PE/ns!)", "data"), np.float32, n_samples_wf),
        (("Width (in ns) of the central 50% area of the peak", "range_50p_area"), np.float32),
        (("Width (in ns) of the central 90% area of the peak", "range_90p_area"), np.float32),
        (("Time between 10% and 50% area quantiles [ns]", "rise_time"), np.float32),
    ]
    return dtype


def _temp_event_data_type(n_samples_wf: int = 150, n_widths: int = 11) -> list:
    """Temp.

    data type which adds field required to use some of the functions used to compute the shape of
    the summed waveform for the TPC.

    """
    dtype = veto_event_waveform_dtype()
    dtype += [
        (
            ("Dummy, total area of all hitlets in event [pe]", "area"),
            np.float32,
        ),
        (
            ("Dummy top waveform data in PE/sample (not PE/ns!)", "data_top"),
            np.float32,
            n_samples_wf,
        ),
        (
            ("Dummy first waveform data in PE/sample (not PE/ns!)", "data_start"),
            np.float32,
            n_samples_wf,
        ),
        (("Weighted average center time of the peak [ns]", "center_time"), np.int64),
        (("Weighted relative median time of the peak [ns]", "median_time"), np.float32),
        (("Peak widths in range of central area fraction [ns]", "width"), np.float32, n_widths),
        (
            ("Peak widths: time between nth and 5th area decile [ns]", "area_decile_from_midpoint"),
            np.float32,
            n_widths,
        ),
    ]

    return dtype
