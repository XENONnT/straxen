import strax
import numpy as np
import numba
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_nv',
        default=(3, 15), track=True,
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'baseline_samples_nv',
        default=10, track=True,
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline'),
    strax.Option(
        'hit_min_amplitude_nv',
        default=20, track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number.'),
)
class nVETOPulseProcessing(strax.Plugin):
    """
    nVETO equivalent of pulse processing. Not much more to say about.
    """
    __version__ = '0.0.5'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'raw_records_coin_nv'
    provides = 'records_nv'
    data_kind = 'records_nv'

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
                       baseline_samples=self.config['baseline_samples_nv'],
                       flip=True)
        strax.integrate(r)

        strax.zero_out_of_bounds(r)

        hits = strax.find_hits(r, min_amplitude=self.config['hit_min_amplitude_nv'])

        le, re = self.config['save_outside_hits_nv']
        r = strax.cut_outside_hits(r, hits, left_extension=le, right_extension=re)
        strax.zero_out_of_bounds(r)

        rlinks = strax.record_links(raw_records_coin_nv)
        r = clean_up_empty_records(r, rlinks, only_last=True)
        return r

@export
@numba.njit(cache=True, nogil=True)
def clean_up_empty_records(records, record_links, only_last=True):
    """
    Function which deletes empty records. Empty means data is completely
    zero.

    :param records: Records which shall be checked.
    :param only_last: If true only last fragments of a pulse are deleted.
    :return: non-empty records

    Note:
        If only_last is false, also records within a pulse can be deleted.
        This may lead to unwanted consequences if it not taken into account.
    """
    indicies_to_keep = np.zeros(len(records), dtype=np.int32)
    n_indicies = 0
    for ind, r in enumerate(records):
        if only_last:
            m_last_fragment = (r['record_i'] > 0) and (r['length'] < len(r['data']))
            if m_last_fragment:
                indicies_to_keep[n_indicies] = ind
                n_indicies += 1
                continue

        if np.any(r['data'] != 0):
            indicies_to_keep[n_indicies] = ind
            n_indicies += 1
            continue

        # If we arrive here this means we will remove this record
        # Hence we have to update the pulse_lengths accordingly:
        length = r['length']

        TRIAL_COUNTER = 500  # Do not want to be stuck forever
        for neighbors in record_links:
            neighbor_i = ind
            ntries = 0

            while ntries < TRIAL_COUNTER:
                neighbor_i = neighbors[neighbor_i]
                if neighbor_i == -1:
                    # No neighbor anymore
                    break
                else:
                    records[neighbor_i]['pulse_length'] -= length
                ntries += 1
            if ntries == TRIAL_COUNTER:
                mes = 'Tried to correct the pulse_length for more than 500 fragments of a single pulse, this is odd.'
                raise TimeoutError(mes)

    return records[indicies_to_keep[:n_indicies]]
