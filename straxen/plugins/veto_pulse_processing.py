import strax
import numpy as np
import numba
import straxen

from straxen.get_corrections import is_cmt_option

export, __all__ = strax.exporter()

MV_PREAMBLE = 'Muno-Veto Plugin: Same as the corresponding nVETO-PLugin.\n'
NV_HIT_OPTIONS = (
    strax.Option(
        'save_outside_hits_nv',
        default=(3, 15), track=True, infer_type=False,
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'hit_min_amplitude_nv', infer_type=False,
        default=('hit_thresholds_nv', 'ONLINE', True), track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number, '
             'or a string like "pmt_commissioning_initial" which means calling '
             'hitfinder_thresholds.py, '
             'or a tuple like (correction=str, version=str, nT=boolean), '
             'which means we are using cmt.'),
)


@export
@strax.takes_config(
    *NV_HIT_OPTIONS,
    strax.Option(
        'baseline_samples_nv', infer_type=False,
        default=('baseline_samples_nv', 'ONLINE', True), track=True,
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline'),
)
class nVETOPulseProcessing(strax.Plugin):
    """
    nVETO equivalent of pulse processing. The following steps are
    applied:

        1. Flip, baseline and integrate waveforms.
        2. Find hits and apply ZLE
        3. Remove empty fragments.
    """
    __version__ = '0.0.8'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'zstd'
    save_when = strax.SaveWhen.TARGET

    depends_on = 'raw_records_coin_nv'
    provides = 'records_nv'
    data_kind = 'records_nv'

    def setup(self):
        if isinstance(self.config['baseline_samples_nv'], int):
            self.baseline_samples = self.config['baseline_samples_nv']
        else:
            self.baseline_samples = straxen.get_correction_from_cmt(
                self.run_id, self.config['baseline_samples_nv'])
        
        # Check config of `hit_min_amplitude_nv` and define hit thresholds
        # if cmt config
        if is_cmt_option(self.config['hit_min_amplitude_nv']):
            self.hit_thresholds = straxen.get_correction_from_cmt(self.run_id,
                self.config['hit_min_amplitude_nv'])
        # if hitfinder_thresholds config
        elif isinstance(self.config['hit_min_amplitude_nv'], str):
            self.hit_thresholds = straxen.hit_min_amplitude(
                self.config['hit_min_amplitude_nv'])
        else: # int or array
            self.hit_thresholds = self.config['hit_min_amplitude_nv']

    def infer_dtype(self):
        record_length = strax.record_length_from_dtype(
            self.deps['raw_records_coin_nv'].dtype_for('raw_records_coin_nv'))
        dtype = strax.record_dtype(record_length)
        return dtype

    def compute(self, raw_records_coin_nv):
        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        r = strax.raw_to_records(raw_records_coin_nv)
        del raw_records_coin_nv

        r = strax.sort_by_time(r)
        strax.zero_out_of_bounds(r)
        strax.baseline(r,
                       baseline_samples=self.baseline_samples,
                       flip=True)

        strax.integrate(r)

        strax.zero_out_of_bounds(r)

        hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)

        le, re = self.config['save_outside_hits_nv']
        r = strax.cut_outside_hits(r, hits, left_extension=le, right_extension=re)
        strax.zero_out_of_bounds(r)

        return r


@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_mv',
        default=(2, 5), track=True, infer_type=False,
        child_option=True, parent_option_name='save_outside_hits_nv',
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'baseline_samples_mv',
        default=100, track=True, infer_type=False,
        child_option=True, parent_option_name='baseline_samples_nv',
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline'),
    strax.Option(
        'hit_min_amplitude_mv', infer_type=False,
        default=('hit_thresholds_mv', 'ONLINE', True), track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_mveto_pmts, or a number, '
             'or a string like "pmt_commissioning_initial" which means calling '
             'hitfinder_thresholds.py, '
             'or a tuple like (correction=str, version=str, nT=boolean),'
             'which means we are using cmt.'),
    strax.Option(
        'check_raw_record_overlaps',
        default=True, track=False, infer_type=False,
        help='Crash if any of the pulses in raw_records overlap with others '
             'in the same channel'),
)
class muVETOPulseProcessing(nVETOPulseProcessing):
    __doc__ = MV_PREAMBLE + nVETOPulseProcessing.__doc__
    __version__ = '0.0.1'
    depends_on = 'raw_records_mv'
    provides = 'records_mv'
    data_kind = 'records_mv'
    child_plugin = True

    def setup(self):
        self.baseline_samples = self.config['baseline_samples_mv']

        # Check config of `hit_min_amplitude_mv` and define hit thresholds
        # if cmt config
        if is_cmt_option(self.config['hit_min_amplitude_mv']):
            self.hit_thresholds = straxen.get_correction_from_cmt(self.run_id,
                self.config['hit_min_amplitude_mv'])
        # if hitfinder_thresholds config
        elif isinstance(self.config['hit_min_amplitude_mv'], str):
            self.hit_thresholds = straxen.hit_min_amplitude(
                self.config['hit_min_amplitude_mv'])
        else: # int or array
            self.hit_thresholds = self.config['hit_min_amplitude_mv']

    def infer_dtype(self):
        record_length = strax.record_length_from_dtype(
            self.deps['raw_records_mv'].dtype_for('raw_records_mv'))
        dtype = strax.record_dtype(record_length)
        return dtype

    def compute(self, raw_records_mv):
        if self.config['check_raw_record_overlaps']:
            straxen.check_overlaps(raw_records_mv, n_channels=3000)
        return super().compute(raw_records_mv)
