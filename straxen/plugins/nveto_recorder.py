import numba
import numpy as np
from scipy.ndimage import convolve1d

import strax
export, __all__ = strax.exporter()

__all__ = ['nVETOrecorder']


@strax.takes_config(
    strax.Option('coincidence_level', type=int, default=4,
                 help="Required coincidence level."),
    strax.Option('resolving_time', type=int, default=302,
                 help="Resolving time of the coincidence in ns.")
)
class nVETORecorder(strax.Plugin):
    __version__ = '0.0.1'
    parallel = 'process'

    rechunk_on_save = False  # same as in tpc pulse_processing
    compressor = 'zstd'  # same as in tpc pulse_processing

    depends_on = 'nveto_raw_records'

    provides = ('nveto_records', 'nveto_diagnostic_records' 'nveto_lone_record_count')

    #     dtype =
    #     data_kind =

    def compute(self, raw_records):
        # First we have to split rr into records r which we would like to store and
        # lone records which we will delet:
        # TODO: Implement sugguestion of Chris Tunnell and save every XX lone hit for
        # DR checks etc.
        r, lr = tight_coincidence(raw_records,
                                  self.config['coincidence_level'],
                                  self.config['resolving_time'])

        # Getting some information about the lr which we will trash:

        # number of lr fragments per channel
        # number of higher order lr fragments
        # total lr area per channel
        # average lr baseline per channel
        # spread of lr baseline per channel

        return {'nveto_records': r,
                'nveto_lone_record_count:' lc}


def count_lone_records(lonerecord):
    """
    Function which estimates some basic properties of the lone
    nveto_records which will not be saved.
    """
    length = len(lonerecord)

    for lr in lonerecord:

        ch = lr['channel']
        area_per_channel[ch] += lr['area']
        n_lr[ch] += 1
        if lr['record_i'] >= 1:
            n_lr_higher_order[ch] += 1
            area_per_channel_higher_order += lr['area']

    return lc


def lone_record_count_dtype(n_channels):
    return [
        (('Lowest start time observed in the chunk', 'time'), np.int64),
        (('Highest endt ime observed in the chunk', 'endtime'), np.int64),
        (('Number of lone records', 'nfragments'),
         (np.int64, n_channels)),
        (('Number of higher order lone fragments', 'nhigherfragments'),
         (np.int64, n_channels)),
        (('Integral of all records in ADC_count x samples', 'lone_record_area'),
         (np.int64, n_channels)),
        (('Integral of lone records in ADC_count x samples', 'lone_pulse_area'),
         (np.int64, n_channels)),
        (('Baseline mean of lone records in ADC_count', 'mean_baseline'),
         (np.int64, n_channels)),
        (('Baseline spread of lone records in ADC_count', 'std_baseline'),
         (np.int64, n_channels))
    ]


# ---------------------
# Auxiliary functions:
# Maybe these function might be useful for other
# plugins as well.
# ----------------------


@numba.njit(nogil=True, cache=True)
def _rr_in_interval(rr, start_times, end_times):
    """
    Function which tests if a raw record is one of the "to be stored"
    intervals.

    Args:
        rr (np.array): raw records
        start_times (np.array): start of the time interval
        end_times (np.array): end of the time interval
    """
    in_window = np.zeros(len(rr), dtype=np.int8)
    indicies = np.arange(len(rr))

    # 2. Now looping over rr and check if they are in the windows:
    for i, r in enumerate(rr):

        # check if raw record is in any interval
        which_interval = (start_times <= r['time']) & (r['time'] <= end_times)
        if np.any(which_interval):
            # tag as to be stored:
            in_window[i] = 1

            # now we have to check if higher fragments exceeds the interval
            # if yes extend interval accordingly:
    #             end_times[which_interval] = np.max([end_times[which_interval],
    #                                                 r['time'] + int(r['length'] * r['dt'])
    #                                                ])

    return in_window


def coincidence(hits, nfold=4, resolving_time=300):
    """
    Checks if n-neighboring events are less apart from each other then
    the specified resolving time.

    Args:
        hits (strax.interval): Can be anything which is of the dtype strax.interval_dtype e.g. records, hits, peaks...
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


def tight_coincidence(data, coincidence_level, resolving_time):
    """
    Tight coincidence for any data_kind which has a 'time' parameter.

    Args:


    returns:
        in_coincidence:
        other:
    """
    intervals = coincidence_intervals(data,
                                      nfold=coincidence_level,
                                      resolving_time=resolving_time)

    mask_in_interval = _rr_in_interval(data, *intervals.T)

    return data[mask_in_interval.astype(bool)], data[np.invert(mask_in_interval.astype(bool))]


def _coincidence(hits, nfold=4, resolving_time=300):
    '''
    Function which checks if n-neighboring events are less apart from each other then
    the specified resolving time.

    Args:
        hits (strax.hits): Found hits, but could be peaks as well.
        nfold (int): coincidence level.
        resolving_time (int): Time window of the coincidence [ns].

    TODO:
        raise an error if dt is not the same among all channels

    returns:
        np.array: array containing the start times of the n-fold coincidence intervals
    '''
    # 1. estimate time difference between leading fragments:
    mask_ri = hits['record_i'] == 0
    start_times = hits[mask_ri]['time']
    pulse_lengths = hits[mask_ri]['length']
    dt = hits['dt'][0]
    # Note sure if dts can be different, but should maybe add a raise here...
    # TODO: Add error condition
    #     if np.sum(hits['dt'] = dt) != len(hits):
    #         raise

    t_diff = diff(start_times)

    # 2. Now we have to check if n-events are within resolving time:
    #   -> use moving average with size n to accumaulate time between n-pulses
    #   -> check if accumulated time is bellow resolving time

    # generate kernal:
    kernal = np.zeros(nfold)
    kernal[:(nfold - 1)] = 1  # weight last seen by t_diff must be zero since
    # starting time point e.g. n=4: [0,1,1,1] --> [f1, f2, f3, f4, ..., fn]

    t_cum = convolve1d(t_diff, kernal, mode='constant', origin=(nfold - 1) // 2)
    t_cum = t_cum[:-(nfold - 1)]  # do not have to check for last < nfold hits

    return start_times[:-(nfold - 1)][t_cum <= resolving_time]


@numba.njit(nogil=True, cache=True)
def _merge_intervals(start_time, resolving_time):
    """
    Function which merges overlapping time intervals into a single one.

    While it is not strictly required, it makes the live easier when saving the raw_records of a higher fragment order.

    Args:
        start_time (np.array): Start time of the different intervals
        resolving_time (int): Coincidence window in ns

    Returns:
        np.array: merged unique time intervals.
    """
    index = np.arange(0, len(start_time), 1)

    # check for gaps larger than 300:
    gaps = diff(start_time) > resolving_time
    gaps[0] = True  # We always need to have index 0
    index = index[gaps]

    # Creating output
    # There are as many gaps as intervals due to gaps[0] = True
    intervals = np.zeros((np.sum(gaps), 2), dtype=np.int64)

    # Looping over all intervals, except for the last one:
    for ind, ci in enumerate(index[1:]):
        end_i = ci - 1
        start_i = index[ind]
        intervals[ind] = [start_time[start_i], start_time[end_i] + resolving_time]

    # Now doing the last one:
    intervals[-1] = [start_time[index[-1]], start_time[len(start_time) - 1] + 300]

    return intervals


@numba.njit(parallel=True, nogil=True)
def diff(array):
    """
    Function which estimates the difference of neighboring values
    in an array.

    The output array has the same shape as the input array.

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
    for i in numba.prange(length):
        res[i + 1] = array[i + 1] - array[i]
    return res
