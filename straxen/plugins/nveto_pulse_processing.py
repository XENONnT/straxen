import numba
import numpy as np

import strax
from straxen import get_to_pe
from straxen import get_resource
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

    depends_on = 'nveto_raw_records'

    provides = 'nveto_records'
    dtype = nveto_pulses_dtype  # Might be the same as records.

    def setup(self):
        self.hit_thresholds = strax.get_resource(self.config['adc_thresholds'], fmt='npy')

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
        return dict(nveto_records=nveto_records)


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
    dtype = nveto_pulses_dtype()

    def setup(self):
        self.to_pe = get_to_pe(self.run_id, self.config['to_pe_file'])
        self.hit_thresholds = get_resource(self.config['adc_thresholds'], fmt='npy')


    def compute(self, nveto_records):
        # Search again for hits in records:
        hits = strax.find_hits(nveto_records, threshold=self.hit_thresholds)

        # Merge overlapping hit boundaries to pulses and sort by time:
        max_channel = np.max(nveto_records['channel'])
        last_hit_in_channel = np.zeros(max_channel,
                                       dtype=[(('Start time of the interval (ns since unix epoch)', 'time'), np.int64),
                                              (('End time of the interval (ns since unix epoch)', 'endtime'), np.int64),
                                              (('Channel/PMT number', 'channel'), np.int16),
                                              (('Time resolution in ns', 'dt'), np.int16)])
        nveto_pulses = concat_overlapping_hits(hits, self.config['nveto_save_outside_hits'], last_hit_in_channel)
        nveto_pulses = strax.sort_by_time(nveto_pulses)

        # Check if hits can be split:


        return dict(nveto_pulses=nveto_pulses)


@export
def nveto_pulses_dtype(n_widths=11):
    return [
        (('Start time of the interval (ns since unix epoch)', 'time'), np.int64),
        (('End time of the interval (ns since unix epoch)', 'endtime'), np.int64),
        (('Channel/PMT number', 'channel'), np.int16),
        (('Time resolution in ns', 'dt'), np.int16),
        (('Area of the PMT pulse in pe', 'area'), np.float64),
        (('Maximum of the PMT pulse in pe/sample', 'height'), np.float64),
        (('Position of the maximum in (ns since unix epoch)', 'amp_time'), np.float64),
        (('Width of the PMT pulse in ns', 'width'), np.float64, n_widths),
        (('Split index 0=No Split, 1=1st part of hit 2=2nd ...', 'split_i'), np.int16),
    ]


@strax.growing_result(nveto_pulses_dtype(), chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def concat_overlapping_hits(hits,
                            extensions,
                            last_hit_in_channel, _result_buffer=None):
    """
    Function which concatenates overlapping hits into a single one.

    Args:
        hits (strax.hits): hits which should be concatenates if necessary.
        extensions (tuple): Tuple containing the left an right extension of the
            hit.
        last_hit_in_channel (np.array): Structure array of the channel length.
            The array must be empty and containing the fields "time", "endtime",
            "channel" and "dt".

    Returns:
        np.array: Array of the pre_nveto_pulses datastructure containing the start
            and end points of the PMT pulses in unix time.

    TODO: Somehow when making last_hit_in_channel a keyword argument
        numba crashes...
    """
    buffer = _result_buffer
    offset = 0

    le, re = extensions

    # Assuming all hits have the same dt:
    last_hit_in_channel['dt'][:] = hits['dt'][0]

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
                res['dt'] = lhc['dt']
                offset += 1

                # Updating current last hit:
                lhc['time'] = st
                lhc['endtime'] = et
                lhc['channel'] = hc


@export
@numba.njit(cache=True, nogil=True)
def get_pulse_data(nveto_records, hit, start_index=0):
    """
    Seraches in a given nveto_record data_chunk for the data a
    specified hit.

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

    Notes:
        For the ussage of start_index function assumes nveto_records are
        sorted in time. (Should also be true for hit(s) if looping over
        them).
    """
    hit_start_time = hit['time']
    hit_end_time = hit['endtime']

    # In case the pulse spans over multiple records we need:
    res_start = 0
    update = True
    found_start = False

    # Init a buffer containing the data:
    nsamples = (hit_end_time - hit_start_time) // nveto_records[0]['dt']
    res = np.zeros(nsamples, dtype=np.int16)

    for index, nvr in enumerate(nveto_records[start_index:], start_index):
        nvr_start_time = nvr['time']
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
            if not found_start:
                start_record = index

            if dt < 0:
                # If this happened our hit should have been in an earlier
                # record.
                # TODO: should we through an error here?
                res[:] = -42
                return res, 0

            found_start = True
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
            return res, start_index


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



        hits = strax.find_hits(nveto_pulses, threshold=self.hit_thresholds)

        # 2. Getting the record data of each hit:


        # Check if hits can be split:

        # Comupte basic properties of the PMT pulses:

        return dict(nveto_pulse_basics=npb)