import strax
import straxen

from straxen.plugins.defaults import MV_PREAMBLE, NV_HIT_DEFAULTS, MV_HIT_DEFAULTS
from straxen.plugins.records_nv.records_nv import nVETOPulseProcessing


export, __all__ = strax.exporter()


@export
class muVETOPulseProcessing(nVETOPulseProcessing):
    __doc__ = MV_PREAMBLE + nVETOPulseProcessing.__doc__
    __version__ = '0.0.1'

    depends_on = 'raw_records_mv'
    provides = 'records_mv'
    data_kind = 'records_mv'
    child_plugin = True

    save_outside_hits_mv = straxen.URLConfig(
        default=MV_HIT_DEFAULTS['save_outside_hits_mv'], track=True, infer_type=False,
        child_option=True, parent_option_name='save_outside_hits_nv',
        help='Save (left, right) samples besides hits; cut the rest')

    baseline_samples_mv = straxen.URLConfig(
        default=100, track=True, infer_type=False,
        child_option=True, parent_option_name='baseline_samples_nv',
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline')

    hit_min_amplitude_mv = straxen.URLConfig(
        infer_type=False,
        default=MV_HIT_DEFAULTS['hit_min_amplitude_mv'], track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_mveto_pmts, or a number, '
             'or a string like "pmt_commissioning_initial" which means calling '
             'hitfinder_thresholds.py, '
             'or a tuple like (correction=str, version=str, nT=boolean),'
             'which means we are using cmt.')

    check_raw_record_overlaps = straxen.URLConfig(
        default=True, track=False, infer_type=False,
        help='Crash if any of the pulses in raw_records overlap with others '
             'in the same channel')

    def setup(self):
        self.baseline_samples = self.baseline_samples_mv
        self.hit_thresholds = self.hit_min_amplitude_mv

    def infer_dtype(self):
        record_length = strax.record_length_from_dtype(
            self.deps['raw_records_mv'].dtype_for('raw_records_mv'))
        dtype = strax.record_dtype(record_length)
        return dtype

    def compute(self, raw_records_mv):
        if self.check_raw_record_overlaps:
            straxen.check_overlaps(raw_records_mv, n_channels=3000)
        return super().compute(raw_records_mv)
