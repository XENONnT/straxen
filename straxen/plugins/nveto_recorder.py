import numba
import numpy as np
from scipy.ndimage import convolve1d
from immutabledict import immutabledict

import strax
import straxen

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('coincidence_level_recorder_nv', type=int, default=3,
                 help="Required coincidence level."),
    strax.Option('pre_trigger_time_nv', type=int, default=150,
                 help="Pretrigger time before coincidence window in ns."),
    strax.Option('resolving_time_recorder_nv', type=int, default=600,
                 help="Resolving time of the coincidence in ns."),
    strax.Option('baseline_samples_nv',
                 default=('baseline_samples_nv', 'ONLINE', True), track=True,
                 help="Number of samples used in baseline rms calculation"),
    strax.Option('hit_min_amplitude_nv',
                 default=20, track=True,
                 help='Minimum hit amplitude in ADC counts above baseline. '
                      'Specify as a tuple of length n_nveto_pmts, or a number.'),
    strax.Option('n_lone_records_nv', type=int, default=2, track=False,
                 help="Number of lone hits to be stored per channel for diagnostic reasons."),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="frozendict mapping subdetector to (min, max) "
                      "channel number."),
    strax.Option('check_raw_record_overlaps_nv',
                 default=False, track=False,
                 help='Crash if any of the pulses in raw_records overlap with others '
                      'in the same channel'),
)
class nVETORecorder(strax.Plugin):
    """
    Plugin which builds a software trigger based on records. The trigger
    is defined by a simple coincidence between pulse fragments within
    the specified resolving time.
    All records which do not fall into one of the coincidence windows
    are labeled as lone records for which we compute some average
    properties for monitoring purposes. Depending on the setting also
    a fixed number of the lone_records per channel are stored.
    """
    __version__ = '0.0.7'
    parallel = 'process'

    rechunk_on_save = True
    save_when = strax.SaveWhen.TARGET
    compressor = 'zstd'

    depends_on = 'raw_records_nv'

    provides = ('raw_records_coin_nv',  # nv-raw-records with coincidence requirement (stored long term)
                'lone_raw_records_nv',
                'lone_raw_record_statistics_nv')

    data_kind = {key: key for key in provides}

    def setup(self):
        if isinstance(self.config['baseline_samples_nv'], int):
            self.baseline_samples = self.config['baseline_samples_nv']
        else:
            self.baseline_samples = straxen.get_correction_from_cmt(
                self.run_id, self.config['baseline_samples_nv'])

    def infer_dtype(self):
        self.record_length = strax.record_length_from_dtype(
            self.deps['raw_records_nv'].dtype_for('raw_records_nv'))

        channel_range = self.config['channel_map']['nveto']
        n_channel = (channel_range[1] - channel_range[0]) + 1
        nveto_records_dtype = strax.raw_record_dtype(self.record_length)
        nveto_diagnostic_lone_records_dtype = strax.record_dtype(self.record_length)
        nveto_lone_records_statistics_dtype = lone_record_statistics_dtype(n_channel)

        dtypes = [nveto_records_dtype,
                  nveto_diagnostic_lone_records_dtype,
                  nveto_lone_records_statistics_dtype]

        return {k: v for k, v in zip(self.provides, dtypes)}

    def compute(self, raw_records_nv, start, end):
        if self.config['check_raw_record_overlaps_nv']:
            straxen.check_overlaps(raw_records_nv, n_channels=3000)
        # Cover the case if we do not want to have any coincidence:
        if self.config['coincidence_level_recorder_nv'] <= 1:
            rr = raw_records_nv
            lr = np.zeros(0, dtype=self.dtype['lone_raw_records_nv'])
            lrs = np.zeros(0, dtype=self.dtype['lone_raw_record_statistics_nv'])
            return {'raw_records_coin_nv': rr,
                    'lone_raw_records_nv': lr,
                    'lone_raw_record_statistics_nv': lrs}

        # Search for hits to define coincidence intervals:
        temp_records = strax.raw_to_records(raw_records_nv)
        temp_records = strax.sort_by_time(temp_records)
        strax.zero_out_of_bounds(temp_records)
        strax.baseline(temp_records,
                       baseline_samples=self.baseline_samples,
                       flip=True)
        hits = strax.find_hits(temp_records,
                               min_amplitude=self.config['hit_min_amplitude_nv'])
        del temp_records


        # First we have to split rr into records and lone records:
        # Please note that we consider everything as a lone record which
        # does not satisfy the coincidence requirement
        intervals = coincidence(hits,
                                self.config['coincidence_level_recorder_nv'],
                                self.config['resolving_time_recorder_nv'],
                                self.config['pre_trigger_time_nv'])
        del hits

        # Always save the first and last resolving_time nanoseconds (e.g. 600 ns)  since we cannot guarantee the gap
        # size to be larger. (We cannot use an OverlapingWindow plugin either since it requires disjoint objects.)
        if len(intervals):
            intervals_with_bounds = np.zeros((len(intervals) + 2, 2), dtype=np.int64)
            intervals_with_bounds[1:-1, :] = intervals
            intervals_with_bounds[0, :] = start, min(start + self.config['resolving_time_recorder_nv'], intervals[0, 0])
            intervals_with_bounds[-1, :] = max(end - self.config['resolving_time_recorder_nv'], intervals[-1, 1]), end
            del intervals
        else:
            intervals_with_bounds = np.zeros((0, 2), dtype=np.int64)

        neighbors = strax.record_links(raw_records_nv)
        mask = pulse_in_interval(raw_records_nv, neighbors, *np.transpose(intervals_with_bounds))
        rr, lone_records = straxen.mask_and_not(raw_records_nv, mask)

        # Compute some properties of the lone_records:
        # We compute only for lone_records baseline etc. since
        # raw_records_nv will be deleted, otherwise we could not change
        # the settings and reprocess the data in case of raw_records_nv
        lr = strax.raw_to_records(lone_records)
        del lone_records

        lr = strax.sort_by_time(lr)
        strax.zero_out_of_bounds(lr)
        strax.baseline(lr,
                       baseline_samples=self.baseline_samples,
                       flip=True)
        strax.integrate(lr)
        lrs, lr = compute_lone_records(lr, self.config['channel_map']['nveto'], self.config['n_lone_records_nv'])
        lrs['time'] = start
        lrs['endtime'] = end

        return {'raw_records_coin_nv': rr,
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

    :param lone_records: raw_records which are flagged as lone
            records.
    :param nveto_channels: First and last channel of nVETO.
    :param n: Number of lone records to be stored per chunk.

    :returns: Structured array of the lone_record_count_dtype
        containing some properties of the lone records which will be
        deleted.
    :returns: Lone records which should be saved for diagnostic purposes.
        The array shape is of the raw_records dtype.
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
def _compute_lone_records(lone_record, res, lone_ids, n, nveto_channels):
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


@numba.njit(cache=True, nogil=True)
def pulse_in_interval(raw_records, record_links, start_times, end_times):
    """
    Checks if a records is in one of the intervals. If yes the entire
    pulse ist flagged as to be stored.

    :param raw_records: raw_records or records
    :param record_links: position of the previous and next record if
        pulse is made of many fragments
    :param start_times: start time of the coincidence intervals
    :param end_times: endtimes of the coincidence intervals
    :return: boolean array true if one fragment of a pulse is in window.
    """
    nrr = len(raw_records)
    result = np.zeros(nrr, np.bool_)

    last_interval_seen = 0
    for ind, rr in enumerate(raw_records):
        # We only have to check the current and the next two intervals:
        st = start_times[last_interval_seen:last_interval_seen + 3]
        et = end_times[last_interval_seen:last_interval_seen + 3]

        if (last_interval_seen + 2) < len(end_times):
            # As soon as we have seen all intervals rr['time'] can be larger
            assert rr['time'] < et[-1], 'This is odd this record omitted the current intervals.'

        # Check if record start is in interval:
        m_starts = rr['time'] >= st
        # <= in m_ends is not ambiguous here since if start and end time of an interval would be the same
        # they would have been merged into a single interval in coincidence.
        m_ends = rr['time'] <= et

        # Check if record end is in interval:
        m_starts = m_starts | (strax.endtime(rr) >= st)
        m_ends = m_ends | (strax.endtime(rr) <= et)
        m = m_starts & m_ends

        if np.any(m):
            # This record is inside one of the interval
            result[ind] = True

            # Update intervals which we have seen already:
            # If we have a funny record for which the start time is in interval 0
            # and the end time in interval 1 we still set last interval seen to be
            # the first interval. It might happen that this record is followed by
            # a second record which is shorter falling only into interval 0. While
            # there will be guaranteed by the definition of our coincidence another
            # record at the start of the interval 1 which will increment last_interval_seen
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
def coincidence(records, nfold=4, resolving_time=300, pre_trigger=0):
    """
    Checks if n-neighboring events are less apart from each other then
    the specified resolving time.

    :param records: Can be anything which is of the dtype
        strax.interval_dtype e.g. records, hits, peaks...
    :param nfold: coincidence level.
    :param resolving_time: Time window of the coincidence [ns].
    :param pre_trigger: Pre trigger window which sould be saved
    :return: array containing the start times and end times of the
            corresponding intervals.

    Note:
        The coincidence window is self-extending. If start times of two
         intervals are exactly resolving_time apart from each other
         they will be merged into a single interval.
    """
    if len(records):
        if nfold > 1:
            start_times = _coincidence(records, nfold, resolving_time)
        else:
            # In case of a "single-fold" coincidence every thing gives
            # the start of a new interval:
            start_times = records['time']
        intervals = _merge_intervals(start_times-pre_trigger, 
                                     resolving_time+pre_trigger)
    else:
        intervals = np.zeros((0, 2), np.int64)
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
        3.) By default we cannot test the last n-1 records since we
            do not know the gap to the next chunk.
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
    # starting time point e.g. n=4: [0,1,1,1] --> [dt1, dt2, dt3, dt4, ..., dtn]  --> 0*dt1 + dt2 + dt3 + dt4

    t_cum = convolve1d(t_diff, kernel, mode='constant', origin=(nfold - 1) // 2)
    # Do not have to check the last n-1 events since by definition they can not satisfy the n-fold coincidence.
    # So we can keep the mask false.
    t_cum = t_cum[:-(nfold - 1)]
    mask[:-(nfold - 1)] = t_cum < resolving_time
    return start_times[mask]


@numba.njit(nogil=True, cache=True)
def _merge_intervals(start_time, resolving_time):
    """
    Function which merges overlapping time intervals into a single one.

    Note:
        If start times of two intervals are exactly resolving_time apart
        from each other they will be merged into a single interval.
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
