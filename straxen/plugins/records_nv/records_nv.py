import strax
import straxen
from straxen.plugins.defaults import NV_HIT_DEFAULTS


export, __all__ = strax.exporter()


@export
class nVETOPulseProcessing(strax.Plugin):
    """
    nVETO equivalent of pulse processing. The following steps are
    applied:

        1. Flip, baseline and integrate waveforms.
        2. Find hits and apply ZLE
        3. Remove empty fragments.
    """

    __version__ = "0.0.8"

    parallel = "process"
    rechunk_on_save = False
    compressor = "zstd"
    save_when = strax.SaveWhen.TARGET

    depends_on = "raw_records_coin_nv"
    provides = "records_nv"
    data_kind = "records_nv"

    save_outside_hits_nv = straxen.URLConfig(
        default=NV_HIT_DEFAULTS["save_outside_hits_nv"],
        track=True,
        infer_type=False,
        help="Save (left, right) samples besides hits; cut the rest",
    )

    hit_min_amplitude_nv = straxen.URLConfig(
        infer_type=False,
        default=NV_HIT_DEFAULTS["hit_min_amplitude_nv"],
        track=True,
        help=(
            "Minimum hit amplitude in ADC counts above baseline. "
            "Specify as a tuple of length n_nveto_pmts, or a number, "
            "or a tuple like (correction=str, version=str, nT=boolean), "
            "which means we are using cmt."
        ),
    )

    baseline_samples_nv = straxen.URLConfig(
        infer_type=False,
        default=26,
        track=True,
        help="Number of samples to use at the start of the pulse to determine the baseline",
    )

    def setup(self):
        self.baseline_samples = self.baseline_samples_nv
        self.hit_thresholds = self.hit_min_amplitude_nv

    def infer_dtype(self):
        record_length = strax.record_length_from_dtype(
            self.deps["raw_records_coin_nv"].dtype_for("raw_records_coin_nv")
        )
        dtype = strax.record_dtype(record_length)
        return dtype

    def compute(self, raw_records_coin_nv):
        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        r = strax.raw_to_records(raw_records_coin_nv)
        del raw_records_coin_nv

        r = strax.sort_by_time(r)
        strax.zero_out_of_bounds(r)
        strax.baseline(r, baseline_samples=self.baseline_samples, flip=True)

        strax.integrate(r)

        strax.zero_out_of_bounds(r)

        hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)

        le, re = self.save_outside_hits_nv
        r = strax.cut_outside_hits(r, hits, left_extension=le, right_extension=re)
        strax.zero_out_of_bounds(r)

        return r
