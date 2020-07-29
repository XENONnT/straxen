import numba
import numpy as np
from scipy.ndimage import convolve1d
from immutabledict import immutabledict

import strax
import straxen

export, __all__ = strax.exporter()

__all__ = ['nVETORecorder']
# TODO: Unify docstrings
@strax.takes_config(
    strax.Option('coincidence_level_recorder_nv', type=int, default=4,
                 help="Required coincidence level."),
    strax.Option('resolving_time_recorder_nv', type=int, default=600,
                 help="Resolving time of the coincidence in ns."),
    strax.Option('nbaseline_samples_lone_records_nv', type=int, default=10, track=False,
                 help="Number of samples used in baseline rms calculation"),
    strax.Option('n_lone_records_nv', type=int, default=2, track=False,
                 help="Number of lone hits to be stored per channel for diagnostic reasons."),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="frozendict mapping subdetector to (min, max) "
                      "channel number."),
    strax.Option('n_nveto_pmts', type=int, track=False,
        help='Number of nVETO PMTs')
)
class nVETORecorder(strax.Plugin):
    __version__ = '0.0.3'
    parallel = 'process'

    rechunk_on_save = True
    compressor = 'lz4'

    depends_on = 'raw_records_prenv'

    provides = ('raw_records_nv', 'lone_raw_records_nv', 'lone_raw_record_statistics_nv')

    data_kind = {key: key for key in provides}

    def infer_dtype(self):
        self.record_length = strax.record_length_from_dtype(
            self.deps['raw_records_prenv'].dtype_for('raw_records_prenv'))

        nveto_records_dtype = strax.raw_record_dtype(self.record_length)
        nveto_diagnostic_lone_records_dtype = strax.record_dtype(self.record_length)
        nveto_lone_records_statistics_dtype = lone_record_statistics_dtype(self.config['n_nveto_pmts'])

        dtypes = [nveto_records_dtype,
                  nveto_diagnostic_lone_records_dtype,
                  nveto_lone_records_statistics_dtype]

        return {k: v for k, v in zip(self.provides, dtypes)}

    def compute(self, raw_records_prenv, start, end):

        strax.zero_out_of_bounds(raw_records_prenv)

        # First we have to split rr into records and lone records:
        # Please note that we consider everything as a lone record which
        # does not satisfy the coincidence requirement
        intervals = coincidence(raw_records_prenv,
                                self.config['coincidence_level_recorder_nv'],
                                self.config['resolving_time_recorder_nv'])
        neighbors = strax.record_links(raw_records_prenv)
        mask = pulse_in_interval(raw_records_prenv, neighbors, *intervals.T)
        rr, lone_records = straxen.mask_and_not(raw_records_prenv, mask)

        # Compute some properties of the lone_records:
        # We compute only for lone_records baseline etc. since
        # pre_raw_records will be deleted, otherwise we could not change
        # the settings and reprocess the data in case of raw_records_nv
        lr = strax.raw_to_records(lone_records)
        del lone_records

        lr = strax.sort_by_time(lr)
        strax.zero_out_of_bounds(lr)
        strax.baseline(lr,
                       baseline_samples=self.config['nbaseline_samples_lone_records_nv'],
                       flip=True)
        strax.integrate(lr)
        lrs, lr = compute_lone_records(lr, self.config['channel_map']['nveto'], self.config['n_lone_records_nv'])
        lrs['time'] = start
        lrs['endtime'] = end
    
        return {'raw_records_nv': rr,
                'lone_raw_records_nv': lr,
                'lone_raw_record_statistics_nv': lrs}


@export
def lone_record_statistics_dtype(n_channels):
    return [
        (('Start time of the chunk', 'time'), np.int64),
        (('Endtime of the chunk', 'endtime'), np.int64),
        (('Channel of the lone record', 'channel'),
         (np.int32, n_channels)),
        (('Total number of lone record fragments', 'nfragments'),
         (np.int32, n_channels)),
        (('Number of higher order lone fragments', 'nhigherfragments'),
         (np.float64, n_channels)),
        (('Average area per waveform in ADC_count x samples', 'lone_record_area'),
         (np.float64, n_channels)),
        (('Average area of higher fragment lone records in ADC_count x samples', 'higher_lone_record_area'),
         (np.int64, n_channels)),
        (('Baseline mean of lone records in ADC_count', 'baseline_mean'),
         (np.float64, n_channels)),
        (('Baseline spread of lone records in ADC_count', 'baseline_rms'),
         (np.float64, n_channels))
    ]


def compute_lone_records(lone_records, nveto_channels, n):
    """
    Function which returns for each data chunk the specified number of
    lone_records and computes for the rest some basic properties.

    Args:
        lone_records (raw_records): raw_records which are flagged as lone
            "hits"
        nveto_channels (tuple): First and last channel of nVETO.
        n (int): Number of lone records which should be stored per data
         chunk.

    Returns:
        numpy.ndarray: Structured array of the lone_record_count_dtype
            containing some properties of the lone records which will be
            deleted.
        numpy.ndarray: Lone records which should be saved for diagnostic
            purposes. The array shape is of the raw_records dtype.
    """
    ch0, ch119 = nveto_channels
    
    if len(lone_records):
        # Results computation of lone records:
        res = np.zeros(1, dtype=lone_record_statistics_dtype(ch119+1-ch0))

        # buffer for lone_records to be stored:
        max_nfrag = np.max(lone_records['record_i'], initial=1)  # We do not know the number of fragments apriori...
        lone_ids = np.ones((ch119 + 1 - ch0, n * max_nfrag), dtype=np.int32) * -1
        _compute_lone_records(lone_records, res[0], lone_ids, n, nveto_channels)
        lone_ids = lone_ids.flatten()
        lone_ids = lone_ids[lone_ids >= 0]
        return res, lone_records[lone_ids]
    return np.zeros(0, dtype=lone_record_statistics_dtype(ch119+1-ch0)), lone_records


@numba.njit(nogil=True, cache=True)
def _compute_lone_records(lone_record, res, lone_ids, n,  nveto_channels):
    ch0, ch119 = nveto_channels
    n_channels = ch119 - ch0 + 1

    # getting start and end time:
    res['time'] = lone_record[0]['time']
    res['endtime'] = lone_record[-1]['time'] + lone_record[-1]['pulse_length'] * lone_record[-1]['dt']
    fragment_max_length = len(lone_record[0]['data'])

    n_lr = np.zeros(n_channels, dtype=np.int32)
    n_index = np.zeros(n_channels, dtype=np.int32)

    for ind, lr in enumerate(lone_record):
        ch = lr['channel']
        ch_ind = ch - ch0
        if n_lr[ch_ind] < n:
            # If we have not found our number of lone_records yet we have to save the index of the event:
            lone_ids[ch_ind][n_index[ch_ind]] = ind  # add event index
            n_index[ch_ind] += 1

            # Check if the event consist out of more than one fragment:
            n_frag = (lr['pulse_length']//fragment_max_length)

            if n_frag and not lr['record_i']:
                # We have to subtract the number of higher fragments from our counter
                n_lr[ch_ind] += 1 - n_frag
            else:
                n_lr[ch_ind] += 1
            continue

        # Computing properties of lone records which will be deleted:
        res['channel'][ch_ind] = ch
        res['lone_record_area'][ch_ind] += lr['area']
        res['nfragments'][ch_ind] += 1
        if lr['record_i'] > 0:
            res['nhigherfragments'][ch_ind] += 1
            res['higher_lone_record_area'][ch_ind] += lr['area']
        else:
            res['baseline_rms'][ch_ind] += lr['baseline_rms']
            res['baseline_mean'][ch_ind] += lr['baseline']

    for ind in range(0, n_channels):
        if res['nfragments'][ind]:
            nwf = res['nfragments'][ind] - res['nhigherfragments'][ind]
            res['baseline_rms'][ind] = res['baseline_rms'][ind] / nwf
            res['baseline_mean'][ind] = res['baseline_mean'][ind] / nwf
            res['lone_record_area'][ind] = res['lone_record_area'][ind] / nwf
        if res['nhigherfragments'][ind]:
            res['higher_lone_record_area'][ind] = res['higher_lone_record_area'][ind] / res['nhigherfragments'][ind]


# ---------------------
# Auxiliary functions:
# Maybe these function might be useful for other
# plugins as well.
# ----------------------
@numba.njit
@numba.njit
def pulse_in_interval(raw_records, record_links, start_times, end_times):
    nrr = len(raw_records)
    result = np.zeros(nrr, np.bool_)

    indicies = np.arange(0, nrr, 1, dtype=np.int32)
    last_interval_seen = 0
    for ind in indicies:
        rr = raw_records[ind]

        # We only have to check the current and next interval:
        m_starts = rr['time'] >= start_times[last_interval_seen:last_interval_seen + 2]
        m_ends = rr['time'] <= end_times[last_interval_seen:last_interval_seen + 2]  # TODO: Is <= ambigious?
        m = m_starts & m_ends

        message = ('This is odd this record omitted the current intervals.' 
                   f'Record time {rr["time"]} last endtime {end_times[last_interval_seen + 2]}')

        assert rr['time'] < end_times[last_interval_seen + 2], message

        if np.any(m):
            # This record is inside an interval
            result[ind] = True

            # Update intervals which we have seen already:
            last_interval_seen = np.argwhere(m)[0, 0] + last_interval_seen

            # Now we have to get all associated records and set them to true:
            for neighbors in record_links:
                n = 0  # some trial counter
                ri = ind
                while n <= 1000:
                    ri = neighbors[ri]
                    if ri == -1:
                        break
                    else:
                        result[ri] = True

                if n == 1000:
                    raise RuntimeWarning('Tried more than 1000 times to find'
                                         ' neighboring record. This is odd.')
    return result


@export
def coincidence(hits, nfold=4, resolving_time=300):
    """
    Checks if n-neighboring events are less apart from each other then
    the specified resolving time.

    Args:
        hits (hits): Can be anything which is of the dtype
            strax.interval_dtype e.g. records, hits, peaks...
        nfold (int): coincidence level.
        resolving_time (int): Time window of the coincidence [ns].
    
    Note:
        The coincidence window is self-extending. The bounds are both
        inclusive. 

    Warning:
        Will not test the last nfold - 1 elements. Since we do not know
        the time of the first element in the next chunk!

    Returns:
        np.array: array containing the start times and end times of the
            corresponding intervals.
    """
    start_times = _coincidence(hits, nfold, resolving_time)
    intervals = _merge_intervals(start_times, resolving_time)

    return intervals


def _coincidence(rr, nfold=4, resolving_time=300):
    """
    Function which checks if n-neighboring events are less apart from
    each other then the specified resolving time.

    Note:
        1.) For the nVETO recorder we treat every fragment as a single
            signal.
        2.) The coincidence window in here is not self extending. Hence
            we compute only the start times of a coincidence window.
        3.) By default we store the last nfold - 1 hits, since there
            might be an overlap to the next chunk. If we can not save
            break the chunks.

    Args:
        rr (raw_records): raw_records
        nfold (int): coincidence level.
        resolving_time (int): Time window of the coincidence [ns].

    Returns:
        np.array: array containing the start times of the n-fold
        coincidence intervals
    """
    # 1. estimate time difference between fragments:
    start_times = rr['time']
    mask = np.zeros(len(start_times), dtype=np.bool_)
    t_diff = np.diff(start_times, prepend=start_times[0])

    # 2. Now we have to check if n-events are within resolving time:
    #   -> use moving average with size n to accumulate time between n-pulses
    #   -> check if accumulated time is below resolving time

    # generate kernel:
    kernel = np.zeros(nfold)
    kernel[:(nfold - 1)] = 1  # weight last seen by t_diff must be zero since
    # starting time point e.g. n=4: [0,1,1,1] --> [f1, f2, f3, f4, ..., fn]

    t_cum = convolve1d(t_diff, kernel, mode='constant', origin=(nfold - 1) // 2)
    t_cum = t_cum[:-(nfold - 1)]  # do not have to check for last < nfold hits
    # since we will store these rr anyhow...
    mask[:-(nfold - 1)] = t_cum <= resolving_time

    return start_times[mask]


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
    gaps = np.diff(start_time) > resolving_time
    
    last_element = np.argwhere(gaps).flatten()
    first_element = 0
    # Creating output
    # There is one more interval than gaps
    intervals = np.zeros((np.sum(gaps) + 1, 2), dtype=np.int64)

    # Looping over all intervals, except for the last one:
    for ind, le in enumerate(last_element):
        intervals[ind] = (start_time[first_element], start_time[le]+resolving_time)
        first_element = le + 1

    # Now we have to deal with the last gap:
    intervals[-1] = (start_time[first_element], start_time[first_element:][-1] + resolving_time)
    return intervals
