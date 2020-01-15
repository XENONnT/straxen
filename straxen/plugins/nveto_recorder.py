import numba
import numpy as np
from scipy.ndimage import convolve1d

import strax
export, __all__ = strax.exporter()

@numba.njit(nogil=True, cache=True)
def _rr_in_interval(rr, start_times, end_times):
    """
    Function which tests if a raw record is one of the "to be stored"
    intervals.

    Args:
        rr (np.array): raw records
        intervals (np.arry): non-overlapping time intervals
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


# ---------------------
# Auxiliary functions:
# Maybe these function might be useful for other
# plugins as well.
# ----------------------

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


def coincidence(hits, nfold=4, resolving_time=300):
    """
    Checks if n-neighboring events are less apart from each other then
    the specified resolving time.

    Args:
        hits (strax.interval): Can be anything which is of the dtype
            strax.interval_dtype e.g. records, hits, peaks...
        nfold (int): coincidence level.
        resolving_time (int): Time window of the coincidence [ns].

    Note:
        A self-extending coincidence window is used here.

    Warnings:
        Due to the different definitions of the 'time' parameter the resulting intervals
        differ between records, hits and peaks!


    returns:
        np.array: array containing the start times and end times of the corresponding
            intervals.
    """

    start_times = _coincidence(hits, nfold, resolving_time)
    intervals = _merge_intervals(start_times, resolving_time)

    return intervals


def _coincidence(hits, nfold=4, resolving_time=300):
    """
    Function which checks if n-neighboring events are less apart from each other then
    the specified resolving time.

    Args:
        hits (strax.hits): Found hits, but could be peaks as well.
        nfold (int): coincidence level.
        resolving_time (int): Time window of the coincidence [ns].

    returns:
        np.array: array containing the start times of the n-fold coincidence intervals
    """
    # 1. estimate time difference between leading fragments:
    mask_ri = hits['record_i'] == 0
    start_times = hits[mask_ri]['time']

    t_diff = diff(start_times)

    # 2. Now we have to check if n-events are within resolving time:
    #   -> use moving average with size n to accumaulate time between n-pulses
    #   -> check if accumulated time is bellow resolving time

    # generate kernal:
    kernal = np.zeros(nfold)
    kernal[:(nfold - 1)] = 1  # weight last seen by t_diff must be zero since
    # starting time point

    t_cum = convolve1d(t_diff, kernal, mode='constant', origin=(nfold - 1) // 2)
    t_cum = t_cum[:-(nfold - 1)]  # do not have to check for last < nfold hits

    return start_times[:-(nfold - 1)][t_cum <= resolving_time]


@numba.njit(nogil=True, cache=True)
def _merge_intervals(start_time, resolving_time):
    """
    Function which merges overlapping time intervals
    into a single one.

    While it is not strictly required, it makes the live easier
    when saving the raw_records of a higher fragment order.

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