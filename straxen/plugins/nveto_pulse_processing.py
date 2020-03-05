import numba
import numpy as np

import strax
import straxen
from straxen import get_to_pe
from straxen import get_resource
export, __all__ = strax.exporter()

__all__ = ['nVETOPulseProcessing', 'nVETOPulseEdges', 'nVETOPulseBasics']


@export
def nveto_pulses_dtype():
    return [
        (('Start time of the interval (ns since unix epoch)', 'time'), np.int64),
        (('End time of the interval (ns since unix epoch)', 'endtime'), np.int64),
        (('Channel/PMT number', 'channel'), np.int16),
        (('Area of the PMT pulse in pe', 'area'), np.float32),
        (('Maximum of the PMT pulse in pe/sample', 'height'), np.float32),
        (('Position of the maximum in (ns since unix epoch)', 'amp_time'), np.int64),
        (('FWHM of the PMT pulse in ns', 'width'), np.float32),
        (('Left edge of the FWHM in ns (minus time)', 'left'), np.float32),
        (('FWTM of the PMT pulse in ns', 'low_width'), np.float32),
        (('Left edge of the FWTM in ns (minus time)', 'low_left'), np.float32),
        (('Split index 0=No Split, 1=1st part of hit 2=2nd ...', 'split_i'), np.int8),
    ]


@export
@strax.takes_config(
    strax.Option(
        'nveto_adc_thresholds',
        default='/dali/lgrandi/wenz/strax_data/HdMtest/find_hits_thresholds.npy',
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

    depends_on = 'nveto_raw_records'
    provides = 'nveto_records'
    data_kind = 'nveto_records'

    dtype = strax.record_dtype(straxen.NVETO_RECORD_LENGTH)  # Might be the same as records.



    def setup(self):
        self.hit_thresholds = get_resource(self.config['nveto_adc_thresholds'], fmt='npy')

    def compute(self, nveto_raw_records):
        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        strax.zero_out_of_bounds(nveto_raw_records)

        hits = strax.find_hits(nveto_raw_records, threshold=self.hit_thresholds)

        le, re = self.config['nveto_save_outside_hits']
        nveto_records = strax.cut_outside_hits(nveto_raw_records, hits, left_extension=le, right_extension=re)

        # Probably overkill, but just to be sure...
        strax.zero_out_of_bounds(nveto_records)

        # Deleting empty data:
        nveto_records = _del_empty(nveto_records, 1)
        return nveto_records


@numba.njit(cache=True, nogil=True)
def _del_empty(records, order=1):
    """
    Function which deletes empty records. Empty means data is completely zero.
    :param records: Records which shall be checked.
    :param order: Fragment order. Cut will only applied to the specified order and
        higher fragments.
    :return: non-empty records
    TODO: Keep track of version in straxen.pulse_processing master
    """
    mask = np.ones(len(records), dtype=np.bool_)
    for ind, r in enumerate(records):
        if r['record_i'] >= order and np.all(r['data'] == 0):
            mask[ind] = False
    return records[mask]


@export
@strax.takes_config(
    strax.Option(
        'nveto_adc_thresholds',
        default='/dali/lgrandi/wenz/strax_data/HdMtest/find_hits_thresholds.npy',
        help='File containing the channel individual hit_finder thresholds.'),
    strax.Option(
        'nveto_save_outside_hits',
        default=(3, 15),
        help='Save (left, right) samples besides hits; cut the rest'),
)
class nVETOPulseEdges(strax.Plugin):
    """
    Plugin which returns the boundaries of the PMT pulses.
    """
    __version__ = '0.0.1'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'nveto_records'
    provides = 'nveto_pulses'
    data_kind = 'nveto_pulses'

    dtype = nveto_pulses_dtype()

    def setup(self):
        self.hit_thresholds = get_resource(self.config['nveto_adc_thresholds'], fmt='npy')

    def compute(self, nveto_records):
        # Search again for hits in records:
        hits = strax.find_hits(nveto_records, threshold=self.hit_thresholds)

        # Merge overlapping hit boundaries to pulses and sort by time:
        max_channel = np.max(nveto_records['channel'])
        last_hit_in_channel = np.zeros(max_channel,
                                       dtype=[(('Start time of the interval (ns since unix epoch)', 'time'), np.int64),
                                              (('End time of the interval (ns since unix epoch)', 'endtime'), np.int64),
                                              (('Channel/PMT number', 'channel'), np.int16)])
        nveto_pulses = concat_overlapping_hits(hits, self.config['nveto_save_outside_hits'], last_hit_in_channel)
        nveto_pulses = strax.sort_by_time(nveto_pulses)

        print('length', len(nveto_pulses), len(nveto_records))
        # Check if hits can be split:
        nvp = split_pulses(nveto_records, nveto_pulses)
        return nvp


@strax.growing_result(nveto_pulses_dtype(), chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def concat_overlapping_hits(hits,
                            extensions,
                            last_hit_in_channel,
                            _result_buffer=None):
    """
    Function which concatenates overlapping hits into a single one.

    Args:
        hits (strax.hits): hits which should be concatenates if necessary.
        extensions (tuple): Tuple containing the left an right extension of the
            hit.
        last_hit_in_channel (np.array): Structure array of the channel length.
            The array must be empty and containing the fields "time", "endtime",
            "channel" and "dt".

    Keyword Args:
        _result_buffer (None): please see strax.growing_result.

    TODO: Somehow when making last_hit_in_channel a keyword argument
        numba crashes...

    Returns:
        np.array: Array of the pre_nveto_pulses data structure containing the start
            and end points of the PMT pulses in unix time.
    """
    buffer = _result_buffer
    offset = 0

    le, re = extensions

    for h in hits:
        st = h['time'] - int(le * h['dt'])
        et = h['time'] + int((h['length'] + re) * h['dt'])
        hc = h['channel']

        lhc = last_hit_in_channel[hc]
        # Have not found any hit in this channel yet:
        if lhc['time'] == 0:
            lhc['time'] = st
            lhc['endtime'] = et
            lhc['channel'] = hc

        # Checking if events overlap:
        else:
            if lhc['endtime'] >= st:
                # Yes, so we have to update only the end_time:
                lhc['endtime'] = et
            else:
                # No, this means we have to save the previous data and update lhc:
                res = buffer[offset]
                res['time'] = lhc['time']
                res['endtime'] = lhc['endtime']
                res['channel'] = lhc['channel']
                offset += 1
                if offset == len(buffer):
                    yield offset
                    offset = 0

                # Updating current last hit:
                lhc['time'] = st
                lhc['endtime'] = et
                lhc['channel'] = hc

    # We went through so now we have to save all remaining hits:
    mask = last_hit_in_channel['time'] != 0
    for lhc in last_hit_in_channel[mask]:
        res = buffer[offset]
        res['time'] = lhc['time']
        res['endtime'] = lhc['endtime']
        res['channel'] = lhc['channel']
        offset += 1
        if offset == len(buffer):
            yield offset
            offset = 0
    yield offset

@export
@numba.njit(cache=True, nogil=True)
def get_pulse_data(nveto_records, hit, start_index=0):
    """
    Searches in a given nveto_record data_chunk for the data a
    specified hit.

    The function will set all samples for which no data can be found
    to -42000..

    Args:
        nveto_records (np.array): Array of the nveto_record_dtype
        hit (np.array): Hit from which the data shall be returned.
            The hit array must contain the fields time, endtime and
            channel.

    Keyword Args:
        start_index (int): Index of the nveto_record from which we should
            start our search.

    Returns:
        np.array: Data of the corresponding hit.
        int: Next start index.
        float: Float part of the baseline for the given event.

    Notes:
        For the usage of start_index function assumes nveto_records are
        sorted in time. (Should also be true for hit(s) if looping over
        them).
    """
    hit_start_time = hit['time']
    hit_end_time = hit['endtime']

    # In case the pulse spans over multiple records we need:
    res_start = 0
    update = True

    # Init a buffer containing the data:
    nsamples = (hit_end_time - hit_start_time) // nveto_records[0]['dt']
    res = np.zeros(nsamples, dtype=np.float32)

    for index, nvr in enumerate(nveto_records[start_index:], start_index):
        nvr_start_time = nvr['time']
        nvr_baseline = nvr['baseline'] % 1
        nvr_length_time = int(nvr['length'] * nvr['dt'])
        nvr_end_time = nvr_start_time + nvr_length_time
        dt = (hit_start_time - nvr_start_time)

        if update and nvr_length_time > dt >= 0:
            start_index = index  # Updating the start_index.
            update = False

        if nvr['channel'] != hit['channel']:
            continue

        # We found the start of our event:
        if dt <= nvr_length_time:
            if dt < 0:
                # If this happend our data or parts of it should have been in an earlier
                # record.
                # TODO: should we throw an error here?
                res[:] = -42000.
                return res, 0, 0.

            start_sample = (hit_start_time - nvr_start_time) // nvr['dt']

            # Start storing the data:
            # Number of samples we found:
            end_sample_time = min(hit_end_time, nvr_end_time)  # Whatever end comes first
            end_sample = (end_sample_time - nvr_start_time) // nvr['dt']
            nsamples_in_fragment = end_sample - start_sample
            res[res_start:res_start + nsamples_in_fragment] = nvr['data'][start_sample:end_sample]

            # Updating the starts in case our record is distributed over more than one fragment:
            res_start += nsamples_in_fragment
            hit_start_time = nvr_end_time

        if res_start == nsamples:
            print('')
            print(res_start, nsamples)
            print('')
            if np.any(res == -42000.):
                print('Nope')
            return res, start_index, nvr_baseline
        else:
            print('WTF', res_start, nsamples)


@strax.growing_result(nveto_pulses_dtype(), chunk_size=int(1e4))
@numba.njit(cache=True, nogil=True)
def split_pulses(records, pulses, _result_buffer=None):
    """
    Function which checks for a given pulse if the pulse should be
    split.

    Note:
        A pulse is split at the local minimum between two maxima if
        one the height difference between one of the maxima and the
        minimum exceeds a certain threshold, or if

    Args:
        records (np.array):
        pulses (np.array):

    Returns:
        np.array

    Notes:
        Function assumes same dt for all channels.
    """
    buffer = _result_buffer
    offset = 0
    record_offset = 0
    dt = records[0]['dt']

    for pulse in pulses:
        # Get data and split pulses:
        print('RO:', record_offset, 'Ch:', pulse['channel'], 'Time:', pulse['time'], 'EndTime:', pulse['endtime'])
        data, record_offset, _ = get_pulse_data(records, pulse, record_offset)
        edges = _split_pulse(data, (0, len(data)))
        edges = edges[edges >= 0]

        # Convert edges into times:
        start_time = pulse['time']
        edges_times = edges * dt + start_time
        # Loop over edges and store them:
        nedges = len(edges_times) - 1
        for ind in range(nedges):
            res = buffer[offset]
            res['time'] = edges_times[ind]
            res['endtime'] = edges_times[ind + 1]
            res['channel'] = pulse['channel']
            if nedges - 1:
                res['split_i'] = ind + 1
            else:
                res['split_i'] = ind  # 0 is reserved for events which were not split.
            offset += 1
            if len(buffer) == offset:
                yield offset
    yield offset


@numba.njit(cache=True, nogil=True)
def _split_pulse(data, edges, min_height=25, min_ratio=0):
    """
    Function which splits the PMT pulses if ncessary.

    Args:
        data (np.array):
        edegs (tuble):

    Keyword Args:
        min_height (int):
        min_ratio (float):

    Returns:
        np.array: Array containing the indicies of the pulses. The
            rest is set to -1. The array has the same length as
            np.diff(edges), since we do not know apprioir the number
            of pulses.
    """
    d = data[edges[0]:edges[1]]
    res = np.ones(len(d), np.int16) * -1  # There cannot be more splot points than sample
    ind = 1
    res[0] = edges[0]
    for split_point in find_split_points(d, min_height=min_height, min_ratio=min_ratio):
        res[ind] = split_point + edges[0]
        ind += 1
    if ind == 1:
        res[ind] = edges[1]
    return res


# -----------------------------------------------
# Taken from split_peaks and adopted to our needs
# Thanks to the author(Jelle?).
# -----------------------------------------------
@numba.jit(nopython=True, nogil=True, cache=True)
def find_split_points(w, min_height=0, min_ratio=0):
    """"Yield indices of prominent local minima in w
    If there was at least one index, yields len(w)-1 at the end
    """
    found_one = False
    last_max = -99999999999999.9
    min_since_max = 99999999999999.9
    min_since_max_i = 0

    for i, x in enumerate(w):
        if x < min_since_max:
            # New minimum since last max
            min_since_max = x
            min_since_max_i = i

        if min(last_max, x) > max(min_since_max + min_height,
                                  min_since_max * min_ratio):
            # Significant local minimum: tell caller,
            # reset both max and min finder
            yield min_since_max_i
            found_one = True
            last_max = x
            min_since_max = 99999999999999.9
            min_since_max_i = i

        if x > last_max:
            # New max, reset minimum finder state
            # Notice this is AFTER the split check,
            # to accomodate very fast rising second peaks
            last_max = x
            min_since_max = 99999999999999.9
            min_since_max_i = i

    if found_one:
        yield len(w)


@export
@strax.takes_config(
    strax.Option(
        'nveto_to_pe_file',
        default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',    # noqa
        help='URL of the to_pe conversion factors'),
)
class nVETOPulseBasics(strax.Plugin):
    """
    nVETO equivalent of pulse processing.
    """
    __version__ = '0.0.1'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = ('nveto_pulses', 'nveto_records')
    provides = 'nveto_pulse_basics'
    dtype = nveto_pulses_dtype()

    def setup(self):
        self.to_pe = get_to_pe(self.run_id, self.config['to_pe_file'])

    def compute(self, nveto_pulses, nveto_records):
        npb = compute_properties(nveto_pulses, nveto_records, self.to_pe)
        return npb


@numba.njit(cache=True, nogil=True)
def compute_properties(nveto_pulses, nveto_records, to_pe):
    """
    Computes the basic PMT pulse properties.

    Args:
        nveto_pulses (np.array): Array of the nveto_pulses_dtype
        nveto_records (np.array): Array of the nveto_records_dtype
        to_pe (np.array): Array containing the gain values of the different
            pmt channels

    Returns:
        np.array: Array of the nveto_pulses_dtype.
    """
    # TODO: Baseline part is not subtracted yet.
    # TODO: Gain stuff is not validated yet.
    dt = nveto_records['dt'][0]
    rind = 0

    for pind, pulse in enumerate(nveto_pulses):
        ch = pulse['channel']

        # parameters to be store:
        area = 0
        height = 0
        amp_ind = 0

        # Getting data and baseline of the event:
        data, rind, b = get_pulse_data(nveto_records, pulse, start_index=rind)

        # Computing area and max bin:
        for ind, d in enumerate(data):
            area += d
            if d > height:
                height = d
                amp_ind = ind
        area = area * dt / to_pe[ch]
        amp_time = pulse['time'] + int(amp_ind * dt)

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

        nvp = nveto_pulses[pind]
        nvp['area'] = area
        nvp['height'] = height
        nvp['amp_time'] = amp_time
        nvp['width'] = width
        nvp['left'] = left_edge
        nvp['low_width'] = width_low
        nvp['low_left'] = left_edge_low
    return nveto_pulses


@numba.njit(cache=True, nogil=True)
def get_fwxm(data, index_maximum, percentage=0.5):
    """
    Estimates the left and right edge of a specific height percentage.

    The function searches for the last sample below and above the specified
    height level on the left and right hand side of the maximum. If the
    samples are found the width is estimated based upon a linear interpolation
    between the samples on the left and right side.
    In case the samples cannot be found for either the right or left hand side
    the correcponding outer bin edges are use: left 0; right last sample + 1.

    Args:
        data (np.array): Data of the pulse.
        index_maximum (ind): Position of the maximum.

    Keyword Args:
        percentage (float): Level for which the witdth shall be computed.

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