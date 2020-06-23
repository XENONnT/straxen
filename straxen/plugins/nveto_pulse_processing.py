import numba
import numpy as np
from immutabledict import immutabledict

import strax
import straxen
export, __all__ = strax.exporter()

__all__ = ['nVETOPulseProcessing', 'nVETOPulseEdges', 'nVETOPulseBasics']


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


pulse_dtype = [(('Start time of the interval (ns since unix epoch)', 'time'), np.int64),
               (('End time of the interval (ns since unix epoch)', 'endtime'), np.int64),
               (('Channel/PMT number', 'channel'), np.int16)]

nveto_pulses_dtype = pulse_dtype + [
        (('Area of the PMT pulse in pe', 'area'), np.float32),
        (('Maximum of the PMT pulse in pe/sample', 'height'), np.float32),
        (('Position of the maximum in (minus time)', 'amp_time'), np.int16),
        (('FWHM of the PMT pulse in ns', 'width'), np.float32),
        (('Left edge of the FWHM in ns (minus time)', 'left'), np.float32),
        (('FWTM of the PMT pulse in ns', 'low_width'), np.float32),
        (('Left edge of the FWTM in ns (minus time)', 'low_left'), np.float32),
        (('Area midpoint in ns relative to pulse start',
          'area_midpoint'), np.float32),
        (('Peak widths in range of central area fraction [ns]',
          'area_decile'), np.float32, 11),
        (('Peak widths: time between nth and 5th area decile [ns]',
          'area_decile_from_midpoint'), np.float32, 11),
        (('Split index 0=No Split, 1=1st part of hit 2=2nd ...', 'split_i'), np.int8)
    ]


@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_nv',
        default=(3, 15),
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'hit_min_amplitude_nv',
        default=20, track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number.'),
    strax.Option(
        'min_split_nv',
        default=25, track=True,
        help='Minimum height difference [ADC counts] between local minimum and maximum, '
             'that a pulse get split.'),
    strax.Option(
        'min_split_ratio_nv',
        default=0, track=True,
        help='Min ratio between local maximum and minimum to split pulse (zero to switch this off).'),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number.")
)
class nVETOPulseEdges(strax.Plugin):
    """
    Plugin which returns the boundaries of the PMT pulses.
    """
    __version__ = '0.0.4'

    parallel = 'process'
    rechunk_on_save = True
    compressor = 'lz4'

    depends_on = 'records_nv'

    provides = 'pulses_nv'
    data_kind = 'pulses_nv'

    dtype = nveto_pulses_dtype

    # def setup(self):
    #     self.hit_thresholds = get_resource(self.config['nveto_adc_thresholds'], fmt='npy')

    def compute(self, records_nv):
        # Search again for hits in records:
        hits = strax.find_hits(records_nv, min_amplitude=self.config['hit_min_amplitude_nv'])

        # Merge overlapping hit boundaries to pulses and sort by time:
        first_channel, last_channel = self.config['channel_map']['nveto']
        nchannels = last_channel - first_channel + 1
        last_hit_in_channel = np.zeros(nchannels,
                                       dtype=[(('Start time of the interval (ns since unix epoch)', 'time'), np.int64),
                                              (('End time of the interval (ns since unix epoch)', 'endtime'), np.int64),
                                              (('Channel/PMT number', 'channel'), np.int16)])
        pulses_nv = concat_overlapping_hits(hits, self.config['save_outside_hits_nv'], last_hit_in_channel, first_channel)
        pulses_nv = strax.sort_by_time(pulses_nv)

        # Check if hits can be split:
        pulses_nv = split_pulses(records_nv,
                                 pulses_nv,
                                 self.config['channel_map']['nveto'],
                                 self.config['min_split_nv'],
                                 self.config['min_split_ratio_nv'])
        return pulses_nv



@export
@strax.takes_config(
    strax.Option(
        'to_pe_file_nv',
        default='/dali/lgrandi/wenz/strax_data/HdMdata_strax_v0_9_0/swt_gains.npy',    # noqa
        help='URL of the to_pe conversion factors. Expect gains in units ADC/sample.'),
    strax.Option(
        'voltage',
        default=2,
        track=True,
        help='Temporal option for digitizer voltage range set during measurement. [V]'),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number.")
)
class nVETOPulseBasics(strax.Plugin):
    """
    nVETO equivalent of pulse processing.
    """
    __version__ = '0.0.4'

    parallel = True
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = ('pulses_nv', 'records_nv')
    provides = 'pulse_basics_nv'

    data_kind = 'pulses_nv'
    dtype = nveto_pulses_dtype

    def setup(self):
        self.to_pe = straxen.get_resource(self.config['to_pe_file_nv'], 'npy')

    def compute(self, pulses_nv, records_nv):
        mvolts_per_adc = self.config['voltage']/2**14*1000
        npb = np.zeros(len(pulses_nv), nveto_pulses_dtype)
        npb = compute_properties(pulses_nv,
                                 records_nv,
                                 self.to_pe,
                                 mvolts_per_adc,
                                 self.config['channel_map']['nveto'],
                                 npb)
        return npb

@export
@numba.njit(cache=True, nogil=True)
def compute_properties(pulses,
                       records,
                       adc_to_pe,
                       volts_per_adc,
                       channels,
                       result_buffer):
    """
    Computes the basic PMT pulse properties.

    Args:
        pulses (np.array): Array of the nveto_pulses_dtype
        records (np.array): Array of the nveto_records_dtype
        adc_to_pe (np.array): Array containing the gain values of the
            different pmt channels. The array should has at least the
            length of max channel + 1.
        volts_per_adc (float): Conversion factor ADC to Volt.

    Returns:
        np.array: Array of the nveto_pulses_dtype.
    """
    # TODO: None of the width estimates is robust against bipolar noise...
    # TODO: Baseline part is not subtracted yet.
    dt = records['dt'][0]
    first_channel, last_channel = channels
    record_offset = np.zeros(last_channel - first_channel + 1, np.int32)

    for pind, pulse in enumerate(pulses):
        # Frequently used quantities:
        ch = pulse['channel']
        t = pulse['time']

        # Getting data and baseline of the event:
        ro = record_offset[ch-first_channel]
        data, ro = get_pulse_data(records, pulse, start_index=ro)
        bl_fpart = records[ro]['baseline'] % 1
        record_offset[ch-first_channel] = ro

        # Computing Amplitude and Amplitude position:
        data = (data + bl_fpart)
        amp_ind = np.argmax(data)
        amp_time = int(amp_ind * dt)
        height = data[amp_ind]

        # Computing FWHM:
        left_edge, right_edge = get_fwxm(data, amp_ind, 0.5)
        left_edge = left_edge * dt + dt / 2
        right_edge = right_edge * dt - dt / 2
        width = right_edge - left_edge

        # Computing FWTM:
        left_edge_low, right_edge = get_fwxm(data, amp_ind, 0.1)
        left_edge_low = left_edge_low * dt + dt / 2
        right_edge = right_edge * dt - dt / 2
        width_low = right_edge - left_edge_low

        # Converting data into PE and compute area and area deciles:
        data = data * adc_to_pe[ch]
        area = np.sum(data)
        area_decils = np.arange(0, 1.01, 0.05)
        pos_deciles = np.zeros(len(area_decils), dtype=np.float32)
        length = len(data)

        # Make some temporal peaks for compute_index_of_fraction
        temp_peak = np.zeros(1, dtype=[(("Integral in PE", 'area'), np.float32),
                                       (('Waveform data in PE/sample', 'data'),
                                        np.int16, length),
                                       (('Length of the interval in samples', 'length'), np.int32)])
        temp_peak['data'] = data
        temp_peak['area'] = area
        temp_peak['length'] = length
        strax.compute_index_of_fraction(temp_peak, area_decils, pos_deciles)


        res = result_buffer[pind]
        res['time'] = t
        res['endtime'] = pulse['endtime']
        res['channel'] = ch
        res['area'] = area
        res['height'] = height * volts_per_adc
        res['amp_time'] = amp_time
        res['width'] = width
        res['left'] = left_edge
        res['low_left'] = left_edge_low
        res['low_width'] = width_low
        res['area_decile'][1:] = (pos_deciles[11:] - pos_deciles[:10][::-1]) * dt
        res['area_midpoint'] = pos_deciles[10] * dt
        res['area_decile_from_midpoint'][:] = (pos_deciles[::2] - pos_deciles[10]) * dt
        res['split_i'] = pulse['split_i']
    return result_buffer



