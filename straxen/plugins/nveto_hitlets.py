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
class nVETOHitlets(strax.Plugin):
    """
    Plugin which returns the boundaries of the PMT pulses.
    """
    __version__ = '0.0.1'

    parallel = 'process'
    rechunk_on_save = True
    compressor = 'lz4'

    depends_on = 'records_nv'

    provides = 'hitlets_nv'
    data_kind = 'hitlets_nv'

    dtype = nveto_pulses_dtype

    # def setup(self):
    #     self.hit_thresholds = get_resource(self.config['nveto_adc_thresholds'], fmt='npy')

    def compute(self, records_nv):
        # Search again for hits in records:
        hits = strax.find_hits(records_nv, min_amplitude=self.config['hit_min_amplitude_nv'])

        # Merge concatenate overlapping  within a channel. This is important
        # in case hits were split by record boundaries. In case we
        # accidentally concatenate two PMT signals we split it later again.
        hits = strax.concat_overlapping_hits(hits,
                                             self.config['save_outside_hits_nv'],
                                             self.config['channel_map']['nveto'])
        hits = strax.sort_by_time(hits)

        # Now convert hits into temp_hitlets including the data field:
        temp_hitlets = np.zeros(len(hits), strax.hitlet_dtype(n_sample=hits['length'].max()))
        strax.refresh_hit_to_hitlets(hits, temp_hitlets)

        # Get hitlet data and split hitlets:
        to_pe = np.repeat(1, 3000)
        strax.get_hitlets_data(temp_hitlets, records, to_pe=to_pe)

        temp_hitlets = strax.split_peaks(temp_hitlets,
                                           records_nv,
                                           np.ones(2120, dtype=np.float32),
                                           data_type='hitlets',
                                           algorithm='local_minimum',
                                           min_height=20)

        # Compute other hitlet properties:

        # Remove data field:

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


@export
@numba.njit(cache=True, nogil=True)
def get_fwxm(data, index_maximum, percentage=0.5):
    """
    Estimates the left and right edge of a specific height percentage.

    Args:
        data (np.array): Data of the pulse.
        index_maximum (ind): Position of the maximum.

    Keyword Args:
        percentage (float): Level for which the width shall be computed.

    Notes:
        The function searches for the last sample below and above the
        specified height level on the left and right hand side of the
        maximum. When the samples are found the width is estimated based
        upon a linear interpolation between the respective samples. In
        case, that the samples cannot be found for either one of the
        sides the corresponding outer most bin edges are used: left 0;
        right last sample + 1.

    Returns:
        float: left edge [sample]
        float: right edge [sample]
    """
    max_val = data[index_maximum]
    max_val = max_val * percentage

    pre_max = data[:index_maximum]
    post_max = data[1 + index_maximum:]

    # First the left edge:
    lbi, lbs = _get_fwxm_boundary(pre_max, max_val)  # coming from the left
    if lbi == -42:
        # We have not found any sample below:
        left_edge = 0.
    else:
        # We found a sample below so lets compute
        # the left edge:
        m = data[lbi + 1] - lbs  # divided by 1 sample
        left_edge = lbi + (max_val - lbs) / m

        # Now the right edge:
    rbi, rbs = _get_fwxm_boundary(post_max[::-1], max_val)  # coming from the right
    if rbi == -42:
        right_edge = len(data)
    else:
        rbi = len(data) - rbi
        m = data[rbi - 2] - rbs
        right_edge = rbi - (max_val - data[rbi - 1]) / m

    return left_edge, right_edge

@export
@numba.njit(cache=True, nogil=True)
def _get_fwxm_boundary(data, max_val):
    """
    Returns sample position and height for the last sample which amplitude is below
    the specified value
    """
    i = -42
    s = -42
    for ind, d in enumerate(data):
        if d < max_val:
            i = ind
            s = d
    return i, s