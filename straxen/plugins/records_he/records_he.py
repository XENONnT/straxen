from immutabledict import immutabledict
import strax
import straxen

export, __all__ = strax.exporter()

from straxen.plugins.defaults import HE_PREAMBLE
from straxen.plugins.records.records import PulseProcessing, pulse_count_dtype


@export
class PulseProcessingHighEnergy(PulseProcessing):
    __doc__ = HE_PREAMBLE + (PulseProcessing.__doc__ or "")
    __version__ = "0.0.1"
    provides = ("records_he", "pulse_counts_he")
    data_kind = {k: k for k in provides}
    rechunk_on_save = immutabledict(records_he=False, pulse_counts_he=True)
    depends_on = "raw_records_he"
    compressor = "zstd"
    child_plugin = True
    save_when = strax.SaveWhen.TARGET

    n_he_pmts = straxen.URLConfig(
        track=False, default=752, infer_type=False, help="Maximum channel of the he channels"
    )

    record_length = straxen.URLConfig(
        default=110, track=False, type=int, help="Number of samples per raw_record"
    )

    hit_min_amplitude_he = straxen.URLConfig(
        default="cmt://hit_thresholds_he?version=ONLINE&run_id=plugin.run_id",
        track=True,
        infer_type=False,
        help="Minimum hit amplitude in ADC counts above baseline. "
        "Specify as a tuple of length n_tpc_pmts, or a number,"
        'or a string like "pmt_commissioning_initial" which means calling'
        "hitfinder_thresholds.py"
        "or a tuple like (correction=str, version=str, nT=boolean),"
        "which means we are using cmt.",
    )

    def infer_dtype(self):
        dtype = dict()
        dtype["records_he"] = strax.record_dtype(self.record_length)
        dtype["pulse_counts_he"] = pulse_count_dtype(self.n_he_pmts)
        return dtype

    def setup(self):
        self.hev_enabled = False

        # FIXME: This looks hacky. Maybe find a better way?
        self.config["n_tpc_pmts"] = self.config["n_he_pmts"]

        self.hit_thresholds = self.hit_min_amplitude_he

    def compute(self, raw_records_he, start, end):
        result = super().compute(raw_records_he, start, end)
        return dict(records_he=result["records"], pulse_counts_he=result["pulse_counts"])
