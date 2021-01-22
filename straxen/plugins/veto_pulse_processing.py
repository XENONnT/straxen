import strax
import numpy as np
import numba
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
    nVETO equivalent of pulse processing. The following steps are
    applied:

        1. Flip, baseline and integrate waveforms.
        2. Find hits and apply ZLE
        3. Remove empty fragments.
    """
    __version__ = '0.0.6'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'raw_records_coin_nv'
    provides = 'records_nv'
    data_kind = 'records_nv'
    ends_with = '_nv'

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

        rlinks = strax.record_links(r)
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
)
class muVETOPulseProcessing(nVETOPulseProcessing):
    __doc__ = MV_PREAMBLE + nVETOPulseProcessing.__doc__
    __version__ = '0.0.1'
    depends_on = 'raw_records_mv'
    provides = 'records_mv'
    data_kind = 'records_mv'
    child_plugin = True

    def infer_dtype(self):
        record_length = strax.record_length_from_dtype(
            self.deps['raw_records_mv'].dtype_for('raw_records_mv'))
        dtype = strax.record_dtype(record_length)
        return dtype

    def compute(self, raw_records_mv):
        return super().compute(raw_records_mv)
