import strax
import numpy as np
import numba
import straxen

from straxen.get_corrections import is_cmt_option

export, __all__ = strax.exporter()

MV_PREAMBLE = 'Muno-Veto Plugin: Same as the corresponding nVETO-PLugin.\n'


@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_nv',
        default=(3, 15), track=True, infer_type=False,
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'baseline_samples_nv', infer_type=False,
        default=('baseline_samples_nv', 'ONLINE', True), track=True,
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline'),
    strax.Option(
        'hit_min_amplitude_nv', infer_type=False,
        default=('hit_thresholds_nv', 'ONLINE', True), track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number, '
             'or a string like "pmt_commissioning_initial" which means calling '
             'hitfinder_thresholds.py, '
             'or a tuple like (correction=str, version=str, nT=boolean), '
             'which means we are using cmt.'),
    strax.Option(
        'min_samples_alt_baseline_nv',
        default=None, track=True, infer_type=False,
        help='Min. length of pulse before alternative baselineing via '
             'pulse median is applied.'),
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

        if self.config['min_samples_alt_baseline_nv']:
            m = r['pulse_length'] > self.config['min_samples_alt_baseline_nv']
            if np.any(m):
                # Correcting baseline after PMT saturated signals
                r[m] = median_baseline(r[m])

        strax.integrate(r)

        strax.zero_out_of_bounds(r)

        hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)

        le, re = self.config['save_outside_hits_nv']
        r = strax.cut_outside_hits(r, hits, left_extension=le, right_extension=re)
        strax.zero_out_of_bounds(r)

        return r


def median_baseline(records):
    """
    Function which computes the baseline according the pulse's median.

    :param records: Records
    """
    # Count number of pulses
    npulses = np.sum(records['record_i'] == 0)
    fail_counter = 0

    if npulses == 1:
        # This case is simple
        records = _correct_baseline(records)
    else:
        # Now the more complicated case in which we have multiple pulses
        # First we have to group our record fragments into their
        # pulses. Hence get record links and group indicies:
        _, nextr = strax.record_links(records)
        pulse_i = []
        # Loop over the begining of every pulse and get all next indicies.
        for i in np.where(records['record_i'] == 0)[0]:
            inds = [i]
            ind = nextr[i]
            # Always look for next index as long there are some
            while ind != -1:
                inds += [ind]
                ind = nextr[ind]
                fail_counter += 1
                assert fail_counter < 5000, 'Stuck in while-loop pulse is longer than 5000 fragments?!?'

            pulse_i.append(inds)

        for pi in pulse_i:
            records[pi] = _correct_baseline(records[pi])
    return records


@numba.njit
def _correct_baseline(records):
    wf = np.zeros(records[0]['pulse_length'], dtype=np.int16)
    for r in records:
        # np.median(records['data']) does not work for numbafied functions
        # Hence we have to get the entire waveforms first
        if r['record_i'] == 0:
            t0 = r['time']
        i = (r['time'] - t0) // r['dt']
        wf[i:i + r['length']] = r['data'][:r['length']]

    bl = np.median(wf)
    for r in records:
        r['data'][:r['length']] = r['data'][:r['length']] - bl
        r['baseline'] -= bl
    return records


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
