import numba
import numpy as np
from scipy.ndimage import convolve1d

import strax
import straxen
from straxen.plugins.pulse_processing import _mask_and_not, channel_split
from strax.processing.pulse_processing import _baseline_rms

export, __all__ = strax.exporter()

__all__ = ['nVETORecorder']


@strax.takes_config(
    strax.Option('coincidence_level', type=int, default=4,
                 help="Required coincidence level."),
    strax.Option('resolving_time', type=int, default=300,
                 help="Resolving time of the coincidence in ns."),
    strax.Option('nbaseline', type=int, default=10,
                 help="Number of samples used in baseline rms calculation"),
    strax.Option('n_lone_hits', type=int, default=1,
                 help="Number of lone hits to be stored per channel for diagnostic reasons. CANNOT BE BELOW 1!"))
class nVETORecorder(strax.Plugin):
    __version__ = '0.0.1'
    parallel = 'process'

    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'nveto_pre_raw_records'

    provides = ('nveto_raw_records', 'nveto_diagnostic_lone_records', 'nveto_lone_records_count')

    data_kind = {key: key for key in provides}

    def infer_dtype(self):
        nveto_records_dtype = strax.record_dtype(straxen.NVETO_RECORD_LENGTH)
        nveto_diagnostic_lone_records_dtype = strax.record_dtype(straxen.NVETO_RECORD_LENGTH)
        nveto_lone_records_count_dtype = lone_record_count_dtype(len(straxen.n_nVETO_pmts))

        dtypes = [nveto_records_dtype,
                  nveto_diagnostic_lone_records_dtype,
                  nveto_lone_records_count_dtype]

        return {k: v for k, v in zip(self.provides, dtypes)}

    def compute(self, nveto_pre_raw_records):
        strax.zero_out_of_bounds(nveto_pre_raw_records)

        # As long as we are working with TPC data we have to split of the diagnostic stuff:
        # TODO split off nVETO acquisition monitor; Note: I think this was moved to the DAQreader
        nveto_pre_raw_records, o = channel_split(nveto_pre_raw_records, straxen.N_PMTS_NVETO)

        # First we have to split rr into records and lone hits:
        intervals = coincidence(nveto_pre_raw_records, self.config['coincidence_level'], self.config['resolving_time'])
        mask = rr_in_interval(nveto_pre_raw_records, *intervals.T)
        raw_records, lone_records = _mask_and_not(nveto_pre_raw_records, mask)

        # Compute some properties of the lone_records:
        lrc, diagnostic_raw_records = compute_lone_records(lone_records, straxen.n_nVETO_pmts,
                                                           self.config['n_lone_hits'],
                                                           self.config['nbaseline'])

        rr = np.zeros(len(raw_records), dtype=strax.record_dtype(straxen.NVETO_RECORD_LENGTH))
        for name in raw_records.dtype.names:
            rr[name][:] = raw_records[name]

        rd = np.zeros(len(diagnostic_raw_records), dtype=strax.record_dtype(straxen.NVETO_RECORD_LENGTH))
        for name in diagnostic_raw_records.dtype.names:
            rd[name][:] = diagnostic_raw_records[name]

        return {'nveto_raw_records': rr,
                'nveto_diagnostic_lone_records': rd,
                'nveto_lone_records_count': lrc}


@export
def lone_record_count_dtype(n_channels):
    return [
        (('Lowest start time observed in the chunk', 'time'), np.int64),
        (('Highest end time observed in the chunk', 'endtime'), np.int64),
        (('Channel of the lone record', 'channel'),
         (np.int32, n_channels)),
        (('Number of lone record fragments', 'nfragments'),
         (np.int32, n_channels)),
        (('Number of higher order lone fragments', 'nhigherfragments'),
         (np.int64, n_channels)),
        (('Integral of all lone records in ADC_count x samples', 'lone_record_area'),
         (np.int64, n_channels)),
        (('Integral of higher fragment lone records in ADC_count x samples', 'higher_lone_record_area'),
         (np.int64, n_channels)),
        (('Baseline mean of lone records in ADC_count', 'baseline_mean'),
         (np.float64, n_channels)),
        (('Baseline spread of lone records in ADC_count', 'baseline_rms'),
         (np.float64, n_channels))
    ]


def compute_lone_records(lone_record, channels, n, nbaseline=10, fragment_max_length=110):
    """
    Function which estimates for each chunk some basic properties of the lone nveto_records.

    Args:
        lone_record (raw_records): raw_records which are flagged as lone "hits"
        channels (numpy.ndarray): List of PMT channels
        n (int): Number of lone records which should be stored per data chunk.

    Keyword Args:
        nbaseline (int): number of samples which is used to compute the baseline rms.

    Returns:
        numpy.ndarray: Structured array of the lone_record_count_dtype containing some
            properties of the lone records which will be deleted.
        numpy.ndarray: Lone records which should be saved for diagnostic purposes. The array shape is of
            the raw_records dtype.
    """
    nchannels = len(channels)

    # Results computation of lone records:
    res = np.zeros(1, dtype=lone_record_count_dtype(nchannels))

    # buffer for lone_records to be stored:
    max_nfrag = np.max(lone_record['record_i'])  # We do not know the number of fragments apriori...
    lone_ids = np.ones((nchannels, n * max_nfrag), dtype=np.int32) * -1

    _compute_lone_records(lone_record, res[0], lone_ids, n, channels, nbaseline,
                          fragment_max_length=fragment_max_length)      # TODO, change this parameter into a config nT

    lone_ids = lone_ids.flatten()
    lone_ids = lone_ids[lone_ids >= 0]

    return res, lone_record[lone_ids]


@numba.njit(nogil=True, cache=True)
def _compute_lone_records(lone_record, res, lone_ids, n,  channels, nbaseline, fragment_max_length=110):
    #TODO: Change boolean indexing into normal indexing, means also change channels argument!

    # getting start and end time:
    res['time'] = lone_record[0]['time']
    res['endtime'] = lone_record[-1]['time']

    channel_index = np.arange(0, len(channels), 1, dtype=np.int16)
    nids = np.zeros(len(channels), dtype=np.int32)

    for ind, lr in enumerate(lone_record):
        ch = lr['channel']
        mask_ch = channels == ch
        ch_i = channel_index[mask_ch][0]

        if nids[ch_i] < n:
            lone_ids[ch_i][nids[ch_i]] = ind  # add event index
            # Check if the event consist out of more fragments:
            n_frag = lr['pulse_length']//fragment_max_length
            nids[ch_i] += 1 - n_frag  # If yes we also have to store them

        # Computing properties of lone records which will be deleted:
        res['channel'][ch_i] = ch
        res['lone_record_area'][ch_i] += lr['area']
        res['nfragments'][ch_i] += 1
        if lr['record_i'] >= 1:
            res['nhigherfragments'][ch_i] += 1
            res['higher_lone_record_area'][ch_i] += lr['area']
        else:
            res['baseline_rms'][ch_i] += _baseline_rms(lr['data'], nbaseline)
            res['baseline_mean'][ch_i] += lr['baseline']

    for ind in range(len(channels)):
        if res['nfragments'][ind]:
                res['baseline_rms'][ind] = (res['baseline_rms'][ind] /
                                            (res['nfragments'][ind] - res['nhigherfragments'][ind]))
                res['baseline_mean'][ind] = (res['baseline_mean'][ind] /
                                             (res['nfragments'][ind] - res['nhigherfragments'][ind]))


# ---------------------
# Auxiliary functions:
# Maybe these function might be useful for other
# plugins as well.
# ----------------------
@numba.njit
def rr_in_interval(rr, start_times, end_times, fragment_max_length=110):
    """
    Function which tests if a raw record is one of the "to be stored"
    intervals.

    Args:
        rr (np.array): raw records
        start_times (np.array): start of the time interval
        end_times (np.array): end of the time interval

    Note:
        Since it might happen that the start_time is defined by a fragment of higher order
        we will also check if the pulse_end is in the to be stored interval.
        We will do the same for higher fragments which may be outside the window. If the start time
        of the very first fragment falls into the window.
    """
    # Here are some example we have to considere:
    # Normal case:
    # Interval:                  |----------------------------------------|
    # If-case:                st___et     st_______et                st_____et
    # Edge cases:
    # Interval:                 |----------------------------------------|
    # else-case      st_______|___et                             st_______|__et
    #             st_____|_____|__et                                 st______|______|__et

    in_window = np.zeros(len(rr), dtype=np.bool_)
    # Looping over rr and check if they are in the windows:
    for i, r in enumerate(rr):
        # TODO: Check if this is faster than calling x times r['...']
        t = r['time']
        dt = r['dt']
        l = r['length']
        pl = r['pulse_length']
        ri = r['record_i']

        # check if current raw_record fragment is in any interval:
        st = t - ri * l * dt                                   # Right edge case
        et = t + int(pl * dt - ri * fragment_max_length * dt)  # Left edge case
        which_interval = (start_times <= st) & (st < end_times) | (start_times <= et) & (et < end_times)
        if np.any(which_interval):
            # tag as to be stored:
            in_window[i] = True
    return in_window


def coincidence(hits, nfold=4, resolving_time=300):
    """
    Checks if n-neighboring events are less apart from each other then
    the specified resolving time.
    Args:
        hits (hits): Can be anything which is of the dtype strax.interval_dtype e.g. records, hits, peaks...
        nfold (int): coincidence level.
        resolving_time (int): Time window of the coincidence [ns].
    Note:
        A self-extending coincidence window is used here.
    Warnings:
        Due to the different interpretations of the 'time' parameter the resulting intervals
        differ between records, hits and peaks!
    Returns:
        np.array: array containing the start times and end times of the corresponding
            intervals.
    """
    start_times = _coincidence(hits, nfold, resolving_time)
    intervals = _merge_intervals(start_times, resolving_time)

    return intervals


def _coincidence(rr, nfold=4, resolving_time=300):
    """
    Function which checks if n-neighboring events are less apart from each other then
    the specified resolving time.
    Note:
        1.) For the nVETO recorder we treat every fragment as a single signal.
        2.) The coincidence window in here is not self extending.
        3.) By default we store the last nfold - 1 hits, since there might be an overlap to the
            next chunk.
    Args:
        rr (raw_records): raw_records
        nfold (int): coincidence level.
        resolving_time (int): Time window of the coincidence [ns].

    Returns:
        np.array: array containing the start times of the n-fold coincidence intervals
    """
    # 1. estimate time difference between fragments:
    start_times = rr['time']
    t_diff = diff(start_times)

    # 2. Now we have to check if n-events are within resolving time:
    #   -> use moving average with size n to accumulate time between n-pulses
    #   -> check if accumulated time is bellow resolving time

    # generate kernel:
    kernel = np.zeros(nfold)
    kernel[:(nfold - 1)] = 1  # weight last seen by t_diff must be zero since
    # starting time point e.g. n=4: [0,1,1,1] --> [f1, f2, f3, f4, ..., fn]

    t_cum = convolve1d(t_diff, kernel, mode='constant', origin=(nfold - 1) // 2)
    t_cum = t_cum[:-(nfold - 1)]  # do not have to check for last < nfold hits

    return start_times[:-(nfold - 1)][t_cum <= resolving_time]


@numba.njit(nogil=True, cache=True)
def _merge_intervals(start_time, resolving_time):
    """
    Function which merges overlapping time intervals into a single one.
    Args:
        start_time (np.array): Start time of the different intervals
        resolving_time (int): Coincidence window in ns
    Returns:
        np.array: merged unique time intervals.
    """
    # check for gaps larger than resolving_time:
    # The gaps will indicate the starts of new intervals
    gaps = diff(start_time) > resolving_time
    interval_starts = np.arange(0, len(gaps), 1)
    interval_starts = interval_starts[gaps]

    # Creating output
    # There is one more interval than gaps
    intervals = np.zeros((np.sum(gaps) + 1, 2), dtype=np.int64)

    # Looping over all intervals, except for the last one:
    for ind, csi in enumerate(interval_starts[:-1]):
        nsi = interval_starts[ind + 1]
        intervals[ind + 1] = [start_time[csi], start_time[nsi - 1] + resolving_time]

    # Now doing the first and last one:
    intervals[0] = [start_time[0], start_time[interval_starts[0] - 1] + resolving_time]
    intervals[-1] = [start_time[interval_starts[-1]], start_time[-1] + resolving_time]

    return intervals


@numba.njit(parallel=True, nogil=True)
def diff(array):
    """
    Function which estimates the difference of neighboring values
    in an array.
    The output array has the same shape as the input array.
    Note:
        The input is expected to be sorted.
    Args:
        array (np.array): array of size n
    returns:
        np.array: array of size n containing the differences between
            two successive values. The leading value is set to 0.
            (Something like array[0] - array [0], just to preserve shape)
    """
    length = len(array)
    res = np.ones(length, dtype=np.int64) * -1
    res[0] = 0
    for i in range(length):
        res[i + 1] = array[i + 1] - array[i]
    return res
