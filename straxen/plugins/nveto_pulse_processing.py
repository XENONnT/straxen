import numba
import numpy as np

import strax
from straxen import get_to_pe
export, __all__ = strax.exporter()



@export
@strax.takes_config(
    strax.Option(
        'nveto_adc_thresholds',
        default='',
        help='File containing the channel individual hit_finder thresholds.'),
    strax.Option(
        'nveto_save_outside_hits',
        default=(3, 15),
        help='Save (left, right) samples besides hits; cut the rest'),
)
class nVETOPulseProcessing(strax.Plugin):
    """
    nVETO equivalent of pulse processing.

    Note:
        I shamelessly copied almost the entire code from the TPC pulse processing. So credit to the
        author of pulse_processing.
    """
    __version__ = '0.0.1'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'nveto_records'

    provides = 'nveto_pulses'
    dtype = nveto_pulses_dtype  # Might be the same as records.

    def setup(self):
        self.hit_thresholds = strax.get_resource(self.config['adc_thresholds'], fmt='npy')

    def compute(self, nveto_records):
        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        strax.zero_out_of_bounds(nveto_records)

        hits = strax.find_hits(nveto_records, threshold=self.hit_thresholds)

        le, re = self.config['nveto_save_outside_hits']
        nveto_pulses = strax.cut_outside_hits(nveto_records, hits, left_extension=le, right_extension=re)

        # Probably overkill, but just to be sure...
        strax.zero_out_of_bounds(nveto_pulses)

        # Deleting empty data:
        nveto_pulses = _del_empty(nveto_pulses, 1)
        return dict(nveto_pulses=nveto_pulses)



@export
@strax.takes_config(
    strax.Option(
        'nveto_to_pe_file',
        default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',    # noqa
        help='URL of the to_pe conversion factors'),
    strax.Option(
        'nveto_adc_thresholds',
        default='',
        help='File containing the channel individual hit_finder thresholds.'),
    strax.Option(
        'nveto_save_outside_hits',
        default=(3, 15),
        help='Save (left, right) samples besides hits; cut the rest'),
)
class nVETOPulseBasics(strax.Plugin):
    """
    nVETO equivalent of pulse processing.
    """
    __version__ = '0.0.1'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'nveto_pulses'

    provides = 'nveto_pulse_basics'
    dtype = nveto_pulse_basic_dtype

    def setup(self):
        self.to_pe = get_to_pe(self.run_id, self.config['to_pe_file'])
        self.hit_thresholds = strax.get_resource(self.config['adc_thresholds'], fmt='npy')


    def compute(self, nveto_pulses):

        # 1. Checking for htis again:
        hits = strax.find_hits(nveto_pulses, threshold=self.hit_thresholds)

        # 2. Getting the record data of each hit:


        # Check if hits can be split:

        # Comupte basic properties of the PMT pulses:

        return dict(nveto_pulse_basics=npb)


def nveto_pulse_basic_dtype(n_widths=11):
    return strax.interval_dtype + [
        (('Area of the PMT pulse in ...', 'area'), np.float64),
        (('Maximum of the PMT pulse in ...', 'height'), np.float64),
        (('Position of the maximum in ...', 'height_pos'), np.float64),
        (('Width of the PMT pulse in ...', 'width'), np.float64),
        (('End time of the interval', 'end_time'), np.float64),
        (('Split index 0=No Split, 1=1st part of hit 2=2nd ...', 'split_i'), np.float64),
    ]

@numba.njit(cache=True, nogil=True)
def _del_empty(records, order=1):
    """
    Function which deletes empty records. Empty means data is completely zero.
    :param records: Records which shall be checked.
    :param order: Fragment order. Cut will only applied to the specified order and
        higher fragments.
    :return: non-empty records
    """
    mask = np.ones(len(records), dtype=np.bool_)
    for ind, r in enumerate(records):
        if r['record_i'] >= order and np.all(r['data'] == 0):
            mask[ind] = False
    return records[mask]