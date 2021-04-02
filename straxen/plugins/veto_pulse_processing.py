import strax
import numpy as np
import numba
import straxen

export, __all__ = strax.exporter()

MV_PREAMBLE = 'Muno-Veto Plugin: Same as the corresponding nVETO-PLugin.\n'


@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_nv',
        default=(3, 15), track=True,
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'baseline_samples_nv',
        default=('baseline_samples_nv', 'ONLINE', True), track=True,
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline'),
    strax.Option(
        'hit_min_amplitude_nv',
        default=20, track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number.'),
    strax.Option(
        'min_samples_alt_baseline_nv',
        default=None, track=True,
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
    ends_with = '_nv'

    def setup(self):
        if isinstance(self.config['baseline_samples_nv'], int):
            self.baseline_samples = self.config['baseline_samples_nv']
        else:
            self.baseline_samples = straxen.get_correction_from_cmt(
                self.run_id, self.config['baseline_samples_nv'])

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

        hits = strax.find_hits(r, min_amplitude=self.config['hit_min_amplitude_nv'])

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
@numba.njit(cache=True, nogil=True)
def clean_up_empty_records(records, record_links, only_last=True):
    """
    Function which deletes empty records. Empty means data is completely
    zero.

    :param records: Records which shall be checked.
    :param record_links: Tuple of previous and next records.
    :param only_last: If true only last fragments of a pulse are deleted.
    :return: non-empty records

    Note:
        If only_last is false, also records within a pulse can be deleted.
        This may lead to unwanted consequences if it not taken into account.
    """
    indicies_to_keep = np.zeros(len(records), dtype=np.int32)
    n_indicies = 0
    for rind, r in enumerate(records):
        if only_last:
            m_first = record_links[0][rind] == strax.NO_RECORD_LINK  #
            m_in_between = record_links[1][rind] != strax.NO_RECORD_LINK
            if m_first or m_in_between:
            # we are not the last record as we don't have a link on the left (i.e. are first) or have a link on the
            # right
                indicies_to_keep[n_indicies] = rind
                n_indicies += 1
                continue

        if np.any(r['data'] != 0):
            indicies_to_keep[n_indicies] = rind
            n_indicies += 1
            continue

        # If we arrive here this means we will remove this record
        # Hence we have to update the pulse_lengths accordingly:
        length = r['length']

        MAX_RECORD_I  = 500  # Do not want to be stuck forever

        left_links, right_links = record_links  # Just to make for loop more explicit
        for ind, neighbors in enumerate((left_links, right_links)):
            if only_last & ind:
                continue

            neighbor_i = rind
            ntries = 0

            # Looping over all left/right neighbors and update pulse_length
            while ntries < MAX_RECORD_I:
                neighbor_i = neighbors[neighbor_i]
                if neighbor_i == strax.NO_RECORD_LINK:
                    # No neighbor anymore
                    break
                else:
                    records[neighbor_i]['pulse_length'] -= length
                ntries += 1
            if ntries == MAX_RECORD_I:
                mes = 'Found more than 500 links for a single pulse this is odd.'

                raise TimeoutError(mes)

    return records[indicies_to_keep[:n_indicies]]


@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_mv',
        default=(2, 5), track=True,
        child_option=True, parent_option_name='save_outside_hits_nv',
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'baseline_samples_mv',
        default=10, track=True,
        child_option=True, parent_option_name='baseline_samples_nv',
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline'),
    strax.Option(
        'hit_min_amplitude_mv',
        default=20, track=True,
        child_option=True, parent_option_name='hit_min_amplitude_nv',
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number.'),
    strax.Option(
        'check_raw_record_overlaps',
        default=True, track=False,
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

    def infer_dtype(self):
        record_length = strax.record_length_from_dtype(
            self.deps['raw_records_mv'].dtype_for('raw_records_mv'))
        dtype = strax.record_dtype(record_length)
        return dtype

    def compute(self, raw_records_mv):
        if self.config['check_raw_record_overlaps']:
            straxen.check_overlaps(raw_records_mv, n_channels=3000)
        return super().compute(raw_records_mv)
