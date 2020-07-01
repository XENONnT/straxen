import strax
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
    __version__ = '0.0.2'

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

    # def setup(self):
    #     self.hit_thresholds = straxen.get_resource(self.config['nveto_adc_thresholds'], fmt='npy')

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
        # TODO: Separate switched off channels for speed up?
        # TODO: Finalize hitfinder threshold. Also has to be done in pulse_edges
        hits = strax.find_hits(r, min_amplitude=self.config['hit_min_amplitude_nv'])

        le, re = self.config['save_outside_hits_nv']
        r = strax.cut_outside_hits(r, hits, left_extension=le, right_extension=re)
        strax.zero_out_of_bounds(r)
        
        # Deleting empty data:
        # TODO: Buggy at the moment fix me:
        # nveto_records = _del_empty(nveto_records, 1)
        return r


# @numba.njit(cache=True, nogil=True)
# def _del_empty(records, order=1):
#     """
#     Function which deletes empty records. Empty means data is completely zero.
#     :param records: Records which shall be checked.
#     :param order: Fragment order. Cut will only applied to the specified order and
#         higher fragments.
#     :return: non-empty records
#     """
#     mask = np.ones(len(records), dtype=np.bool_)
#     for ind, r in enumerate(records):
#         if r['record_i'] >= order and np.all(r['data'] == 0):
#             mask[ind] = False
#     return records[mask]



