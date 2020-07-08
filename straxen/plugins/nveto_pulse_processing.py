import strax
import numpy as np
export, __all__ = strax.exporter()

__all__ = ['nVETOPulseProcessing']


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
    nVETO equivalent of pulse processing.

    Note:
        I shamelessly copied almost the entire code from the TPC pulse processing. So credit to the
        author of pulse_processing.
    """
    __version__ = '0.0.3'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'raw_records_nv'
    provides = 'records_nv'
    data_kind = 'records_nv'

    def infer_dtype(self):
        record_length = strax.record_length_from_dtype(
            self.deps['raw_records_nv'].dtype_for('raw_records_nv'))
        dtype = strax.record_dtype(record_length)
        return dtype

    def compute(self, raw_records_nv):
        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        r = strax.raw_to_records(raw_records_nv)
        del raw_records_nv

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

        r = clean_up_empty_records(r, allow_all=False)
        return r

@numba.njit(cache=True, nogil=True)
def clean_up_empty_records(records, allow_all=False):
    """
    Function which deletes empty records. Empty means data is completely
    zero.

    :param records: Records which shall be checked.
    :param allow_all: If true allows to delete intermediate fragments
        which are between two records.
    :return: non-empty records
    """
    indicies = np.zeros(len(records), dtype=np.bool_)
    n_indicies = 0
    for ind, r in enumerate(records):
        m_last_fragment = (r['record_i'] > 0) and (r['length'] < len(r['data']))
        if not allow_all and m_last_fragment:
            continue
        if np.all(r['data'] == 0):
            indicies[n_indicies] = ind
            n_indicies += 1
    return records[indicies[:n_indicies]]


