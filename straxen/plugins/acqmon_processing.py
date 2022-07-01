import typing
from enum import IntEnum
import numba
import numpy as np
import strax
import straxen
from immutabledict import immutabledict

from .daqreader import ARTIFICIAL_DEADTIME_CHANNEL

export, __all__ = strax.exporter()


# More info about the acquisition monitor can be found here:
# https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:alexelykov:acquisition_monitor

class AqmonChannels(IntEnum):
    """Mapper of named aqmon channels to ints"""
    MV_TRIGGER = 797
    GPS_SYNC = 798
    ARTIFICIAL_DEADTIME = ARTIFICIAL_DEADTIME_CHANNEL
    # Analogue sum waveform
    SUM_WF = 800
    # GPS sync acquisition monitor
    GPS_SYNC_AM = 801
    # HighEnergyVeto
    HEV_STOP = 802
    HEV_START = 803
    # To avoid confusion with HEV, these are the high energy boards
    BUSY_HE_STOP = 804
    BUSY_HE_START = 805
    # Low energy boards (main chain)
    BUSY_STOP = 806
    BUSY_START = 807
    # nVETO GPS_Sync
    GPS_SYNC_NV = 813
    # mVETO GPS_Sync
    GPS_SYNC_MV = 1084


@export
class AqmonHits(strax.Plugin):
    """
    Find hits in acquisition monitor data. These hits could be
    then used by other plugins for deadtime calculations,
    GPS SYNC analysis, etc.
    """
    save_when = strax.SaveWhen.TARGET
    __version__ = '1.1.3'
    hit_min_amplitude_aqmon = straxen.URLConfig(
        default=(
            # Analogue signals
            (50, (int(AqmonChannels.SUM_WF),)),
            # Digital signals, can set a much higher threshold
            (1500, (
                int(AqmonChannels.MV_TRIGGER),
                int(AqmonChannels.GPS_SYNC),
                int(AqmonChannels.GPS_SYNC_AM),
                int(AqmonChannels.HEV_STOP),
                int(AqmonChannels.HEV_START),
                int(AqmonChannels.BUSY_HE_STOP),
                int(AqmonChannels.BUSY_HE_START),
                int(AqmonChannels.BUSY_STOP),
                int(AqmonChannels.BUSY_START),)),
            # Fake signals, 0 meaning that we won't find hits using
            # strax but just look for starts and stops
            (0, (int(AqmonChannels.ARTIFICIAL_DEADTIME),)),

        ),
        track=True,
        help='Minimum hit threshold in ADC*counts above baseline. Specified '
             'per channel in the format (threshold, (chx,chy),)',
    )
    baseline_samples_aqmon = straxen.URLConfig(
        default=10,
        track=True,
        help='Number of samples to use at the start of the pulse to determine the baseline'
    )
    check_raw_record_aqmon_overlaps = straxen.URLConfig(
        default=True,
        track=False,
        help='Crash if any of the pulses in raw_records_aqmon overlap with others '
             'in the same channel'
    )
    wanted_fields = straxen.URLConfig(
        default=['time', 'length', 'dt', 'channel', 'area'],
        track=False,
        help='Fields of hits that are actually needed, in order to save a little '
             'disk space from unused zeros.'
    )

    depends_on = 'raw_records_aqmon'
    provides = 'aqmon_hits'
    data_kind = 'aqmon_hits'

    dtype = [dt for dt in strax.hit_dtype if dt[0][1] in wanted_fields]


    def compute(self, raw_records_aqmon):
        not_allowed_channels = (set(np.unique(raw_records_aqmon['channel']))
                                - set(self.aqmon_channels))
        if not_allowed_channels:
            raise ValueError(
                f'Unknown channel {not_allowed_channels}. Only know {self.aqmon_channels}')

        if self.check_raw_record_aqmon_overlaps:
            straxen.check_overlaps(raw_records_aqmon,
                                   n_channels = max(AqmonChannels).value + 1
            )

        records = strax.raw_to_records(raw_records_aqmon)
        strax.zero_out_of_bounds(records)
        strax.baseline(records, baseline_samples=self.baseline_samples_aqmon, flip=True)
        aqmon_hits = self.find_aqmon_hits_per_channel(records)
        aqmon_hits = strax.sort_by_time(aqmon_hits)

        # in busy runs veto starts get split at chunk edges into two. we introduce hitlets:
        aqmon_hitlets = strax.create_hitlets_from_hits(aqmon_hits, (0,0), [0, 1000])
        to_pe = np.ones(808)  # stay in ADC units: these are NIM signals
        aqmon_hitlets = strax.get_hitlets_data(aqmon_hitlets, records, to_pe=to_pe)

        return aqmon_hitlets[self.wanted_fields]

    @property
    def aqmon_channels(self):
        return [channel for hit_and_channel_list in self.hit_min_amplitude_aqmon
                for channel in hit_and_channel_list[1]]

    def find_aqmon_hits_per_channel(self, records):
        """Allow different thresholds to be applied to different channels"""
        aqmon_thresholds = np.zeros(np.max(self.aqmon_channels) + 1)
        for hit_threshold, channels in self.hit_min_amplitude_aqmon:
            aqmon_thresholds[np.array(channels)] = hit_threshold

        # Split the artificial deadtime ones and do those separately if there are any
        is_artificial = records['channel'] == AqmonChannels.ARTIFICIAL_DEADTIME
        aqmon_hits = strax.find_hits(records[~is_artificial],
                                     min_amplitude=aqmon_thresholds)

        if np.sum(is_artificial):
            aqmon_hits = np.concatenate([
                aqmon_hits, self.get_deadtime_hits(records[is_artificial])])
        return aqmon_hits

    def get_deadtime_hits(self, artificial_deadtime):
        """
        Actually, the artificial deadtime hits are already an interval so
        we only have to copy the appropriate hits
        """
        hits = np.zeros(len(artificial_deadtime), dtype=self.dtype)
        hits['time'] = artificial_deadtime['time']
        hits['dt'] = artificial_deadtime['dt']
        hits['length'] = artificial_deadtime['length']
        hits['channel'] = artificial_deadtime['channel']
        return hits


# ### Veto hardware ###:
# V1495 busy veto module:
# Generates a 25 ns NIM pulse whenever a veto begins and a 25 ns NIM signal when it ends.
# A new start signal can occur only after the previous busy instance ended.
# 1ms (1e6 ns) - minimum busy veto length, or until the board clears its memory

# DDC10 High Energy Veto:
# 10ms (1e7 ns) - fixed HE veto length in XENON1T DDC10,
# in XENONnT it will be calibrated based on the length of large S2 SE tails
# The start/stop signals for the HEV are generated by the V1495 board


@export
class VetoIntervals(strax.OverlapWindowPlugin):
    """ Find pairs of veto start and veto stop signals and the veto
    duration between them:
     - busy_*  <= V1495 busy veto for tpc channels
     - busy_he_*    <= V1495 busy veto for high energy tpc channels
     - hev_*   <= DDC10 hardware high energy veto
     - straxen_deadtime <= special case of deadtime introduced by the
       DAQReader-plugin
    """
    __version__ = '1.1.0'
    depends_on = 'aqmon_hits'
    provides = 'veto_intervals'
    data_kind = 'veto_intervals'

    # This option is just showing where the OverlapWindowPlugin fails.
    # We need to buffer the entire run in order not to run into chunking
    # issues. A better solution would be using
    #   github.com/AxFoundation/strax/pull/654
    max_veto_window = straxen.URLConfig(
        default=int(7.2e12),
        track=True,
        type=int,
        help='Maximum separation between veto stop and start pulses [ns]. '
             'Set to be >> than the max duration of the run to be able to '
             'fully store one run into buffer since aqmon-hits are not '
             'sorted by endtime'
    )

    def infer_dtype(self):
        dtype = [(('veto interval [ns]', 'veto_interval'), np.int64),
                 (('veto signal type', 'veto_type'), np.str_('U20'))]
        dtype += strax.time_fields
        return dtype

    def setup(self):
        self.veto_names = ['busy_', 'busy_he_', 'hev_']
        self.channel_map = {aq_ch.name.lower(): int(aq_ch)
                            for aq_ch in AqmonChannels}

    def get_window_size(self):
        # Give a very wide window
        return self.max_veto_window

    def compute(self, aqmon_hits, start, end):
        # Allocate a nice big buffer and throw away the part we don't need later
        result = np.zeros(len(aqmon_hits) * len(self.veto_names), self.dtype)
        vetos_seen = 0

        for veto_name in self.veto_names:
            veto_hits_start = channel_select(aqmon_hits, self.channel_map[veto_name + 'start'])
            veto_hits_stop = channel_select(aqmon_hits, self.channel_map[veto_name + 'stop'])

            veto_hits_start, veto_hits_stop = self.handle_starts_and_stops_outside_of_run(
                veto_hits_start=veto_hits_start,
                veto_hits_stop=veto_hits_stop,
                chunk_start=start,
                chunk_end=end,
                veto_name=veto_name,
            )
            n_vetos = len(veto_hits_start)

            result["time"][vetos_seen:vetos_seen + n_vetos] = veto_hits_start['time']
            result["endtime"][vetos_seen:vetos_seen + n_vetos] = veto_hits_stop['time']
            result["veto_type"][vetos_seen:vetos_seen + n_vetos] = veto_name + 'veto'

            vetos_seen += n_vetos

        # Straxen deadtime is special, it's a start and stop with no data
        # but already an interval so easily used here
        artificial_deadtime = aqmon_hits[(aqmon_hits['channel'] ==
                                          AqmonChannels.ARTIFICIAL_DEADTIME)]
        n_artificial = len(artificial_deadtime)

        if n_artificial:
            result[vetos_seen:n_artificial]['time'] = artificial_deadtime['time']
            result[vetos_seen:n_artificial]['endtime'] = strax.endtime(artificial_deadtime)
            result[vetos_seen:n_artificial]['veto_type'] = 'straxen_deadtime_veto'
            vetos_seen += n_artificial

        result = result[:vetos_seen]
        result['veto_interval'] = result['endtime'] - result['time']
        sort = np.argsort(result['time'])
        result = result[sort]
        return result

    def handle_starts_and_stops_outside_of_run(
            self,
            veto_hits_start: np.ndarray,
            veto_hits_stop: np.ndarray,
            chunk_start: int,
            chunk_end: int,
            veto_name: str,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        We might be missing one start or one stop at the end of the run,
        set it to the chunk endtime if this is the case
        """
        # Just for traceback info that we declare this here
        extra_start = []
        extra_stop = []
        missing_a_final_stop = (
                len(veto_hits_start)
                and len(veto_hits_stop)
                and veto_hits_start[-1]['time'] > veto_hits_stop['time'][-1])
        missing_a_final_stop = (
                missing_a_final_stop
                or (len(veto_hits_start) and not len(veto_hits_stop)
                    )
        )
        if missing_a_final_stop:
            # There is one *start* of the //end// of the run -> the
            # **stop** is missing (because it's outside of the run),
            # let's add one **stop** at the //end// of this chunk
            extra_stop = self.fake_hit(chunk_end)
            veto_hits_stop = np.concatenate([veto_hits_stop, extra_stop])
        if len(veto_hits_stop) - len(veto_hits_start) == 1:
            # There is one *stop* of the //beginning// of the run
            # -> the **start** is missing (because it's from before
            # starting the run), # let's add one **start** at the
            # //beginning// of this chunk
            extra_start = self.fake_hit(chunk_start)
            veto_hits_start = np.concatenate([extra_start, veto_hits_start])

        something_is_wrong = len(veto_hits_start) != len(veto_hits_stop)

        message = (f'Got inconsistent number of {veto_name} starts '
                   f'{len(veto_hits_start)}) / stops ({len(veto_hits_stop)}).')
        if len(extra_start):
            message += ' Despite the fact that we inserted one extra start at the beginning of the run.'  # noqa
        elif len(extra_stop):
            message += ' Despite the fact that we inserted one extra stop at the end of the run.'  # noqa
        if something_is_wrong:
            raise ValueError(message)

        if np.any(veto_hits_start['time'] > veto_hits_stop['time']):
            raise ValueError('Found veto\'s starting before the previous stopped')

        return veto_hits_start, veto_hits_stop

    @staticmethod
    def fake_hit(start, dt=1, length=1):
        hit = np.zeros(1, strax.hit_dtype)
        hit['time'] = start
        hit['dt'] = dt
        hit['length'] = length
        return hit


# Don't use @numba since numba doesn't like masking arrays, use numpy
def channel_select(rr, ch):
    """Return data from start/stop veto channel in the acquisition monitor (AM)"""
    return rr[rr['channel'] == ch]


@export
class VetoProximity(strax.OverlapWindowPlugin):
    """
    Find the closest next/previous veto start w.r.t. the event time or
    when a busy happens during an event.
    """

    __version__ = '2.1.0'
    # Strictly speaking, we could depend on 'events', but then you couldn't
    # change the event_window_fields to e.g. s1_time and s2_endtime.
    depends_on = ('event_basics', 'veto_intervals')
    provides = 'veto_proximity'
    data_kind = 'events'

    event_window_fields = straxen.URLConfig(
        default=('time', 'endtime'),
        help='Fields to determine where to look for overlaps for using '
             'this plugin in the events. The default uses start and endtime '
             'of an event, but this can also be the S1 or S2 start/endtime'
    )

    veto_proximity_window = straxen.URLConfig(
        default=int(300e9),
        help='Maximum separation between veto stop and start pulses [ns]'
    )
    time_no_aqmon_veto_found = straxen.URLConfig(
        default=int(3.6e+12),
        track=True,
        type=int,
        help='If no next/previous veto is found, we will fill the fields '
             'time_to_previous_XX with this time. Set to a large number '
             'such that one will never cut events that are < YY ns.'
    )

    veto_names = ['busy', 'busy_he', 'hev', 'straxen_deadtime']

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields
        start_field, stop_field = self.event_window_fields
        for name in self.veto_names:
            dtype += [
                ((f'Duration of event overlapping with "{name}"-veto [ns]',
                  f'veto_{name}_overlap'),
                 np.int64),
                ((f'Time (absolute value) to previous "{name}"-veto '
                  f'from "{start_field}" of event [ns]',
                  f'time_to_previous_{name}'),
                 np.int64),
                ((f'Time (absolute value) to next "{name}"-veto '
                  f'from "{stop_field}" of event [ns]',
                  f'time_to_next_{name}'),
                 np.int64),
            ]

        return dtype

    def get_window_size(self):
        return self.veto_proximity_window

    def set_result_for_veto(self,
                            result_buffer: np.ndarray,
                            event_window: np.ndarray,
                            veto_intervals: np.ndarray,
                            veto_name: str) -> None:
        """
        Fill the result buffer inplace. Goal is to find vetos with
        <veto_name> that are either during, before or after the
         current event_window.

        :param result_buffer: The buffer to fill inplace
        :param event_window: start/stop boundaries of the event to consider.
            Should be an array with ['time'] and ['endtime'] which can be
            based on event start/end times or S1/S2 times
        :param veto_intervals: veto intervals datatype
        :param veto_name: The name of the veto to fill the result buffer for
        :return: Nothing, results are filled in place
        """
        # Set defaults to be some very long time
        result_buffer[f'time_to_previous_{veto_name}'] = self.time_no_aqmon_veto_found
        result_buffer[f'time_to_next_{veto_name}'] = self.time_no_aqmon_veto_found

        selected_intervals = veto_intervals[veto_intervals['veto_type'] == f'{veto_name}_veto']
        if not len(selected_intervals):
            return

        vetos_during_event = strax.touching_windows(selected_intervals,
                                                    event_window)

        # Figure out the vetos *during* an event
        for event_i, veto_window in enumerate(vetos_during_event):
            if veto_window[1] - veto_window[0]:
                vetos_in_window = selected_intervals[veto_window[0]:
                                                     veto_window[1]].copy()
                starts = np.clip(vetos_in_window['time'],
                                 event_window[event_i]['time'],
                                 event_window[event_i]['endtime'])
                stops = np.clip(vetos_in_window['endtime'],
                                event_window[event_i]['time'],
                                event_window[event_i]['endtime'])
                # Now sum over all the stops-starts that are clipped
                # within the duration of the event
                result_buffer[event_i][f'veto_{veto_name}_overlap'] = np.sum(stops -
                                                                             starts)

        # Find the next and previous veto's
        times_to_prev, times_to_next = self.abs_time_to_prev_next(event_window, selected_intervals)
        mask_prev = times_to_prev > 0
        result_buffer[f'time_to_previous_{veto_name}'][mask_prev] = times_to_prev[mask_prev]

        max_next = times_to_next > 0
        result_buffer[f'time_to_next_{veto_name}'][max_next] = times_to_next[max_next]

    @staticmethod
    @numba.njit
    def abs_time_to_prev_next(event_window, selected_intervals):
        """Get the absolute time to the previous and the next interval"""
        times_to_prev = np.ones(len(event_window)) * -1
        times_to_next = np.ones(len(event_window)) * -1
        for event_i, ev_wind in enumerate(event_window):
            # Two cases left, either veto's are before or after the event window
            interval_before = selected_intervals['endtime'] < ev_wind['time']
            interval_after = selected_intervals['time'] > ev_wind['endtime']

            if np.sum(interval_before):
                prev_intervals = selected_intervals[interval_before]
                time_to_prev = np.abs(ev_wind['time'] - prev_intervals['endtime'])
                prev_idx = np.argmin(time_to_prev)
                times_to_prev[event_i] = time_to_prev[prev_idx]

            if np.sum(interval_after):
                next_intervals = selected_intervals[interval_after]
                time_to_next = np.abs(next_intervals['endtime'] - ev_wind['endtime'])
                next_idx = np.argmin(time_to_next)
                times_to_next[event_i] = time_to_next[next_idx]
        return times_to_prev, times_to_next

    def compute(self, events, veto_intervals):
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']

        # Get containers for touching windows based on self.event_window_fields
        event_window = np.zeros(len(events), dtype=strax.time_fields)
        event_window['time'] = events[self.event_window_fields[0]]
        event_window['endtime'] = events[self.event_window_fields[1]]

        for veto_name in self.veto_names:
            self.set_result_for_veto(result, event_window, veto_intervals, veto_name)
        return result


class DetectorSynchronization(strax.Plugin):
    """
    Plugin which computes the synchronization delay between TPC and
    vetos.

    Reference:
        * xenon:xenonnt:dsg:mveto:sync_monitor
    """
    __version__ = '0.0.3'
    depends_on = ('raw_records_aqmon',
                  'raw_records_aqmon_nv',
                  'raw_records_aux_mv')
    provides = 'detector_time_offsets'
    data_kind = 'detector_time_offsets'

    tpc_internal_delay = straxen.URLConfig(
        default={'0': 4917, '020380': 10137},
        type=dict,
        track=True,
        help='Internal delay between aqmon and regular TPC channels ins [ns]'
    )
    adc_threshold_nim_signal = straxen.URLConfig(
        default=500,
        type=int,
        track=True,
        help='Threshold in [adc] to search for the NIM signal'
    )
    # This value is only valid for SR0:
    epsilon_offset = straxen.URLConfig(
        default=76,
        type=int,
        track=True,
        help='Measured missing offset for nveto in [ns]'
    )
    sync_max_delay = strax.Config(
        default=11e3,
        help='max delay DetectorSynchronization [ns]')
    sync_expected_min_clock_distance = straxen.URLConfig(
        default=9.9e9,
        help='min clock distance DetectorSynchronization [ns]')
    sync_expected_max_clock_distance = straxen.URLConfig(
        default=10.1e9,
        help='max clock distance DetectorSynchronization [ns]')

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields
        dtype += [(('Time offset for nV to synchronize with TPC in [ns]',
                    'time_offset_nv'), np.int64),
                  (('Time offset for mV to synchronize with TPC in [ns]',
                    'time_offset_mv'), np.int64)
                  ]
        return dtype

    def compute(self,
                raw_records_aqmon,
                raw_records_aqmon_nv,
                raw_records_aux_mv,
                start, end):
        rr_tpc = raw_records_aqmon
        rr_nv = raw_records_aqmon_nv
        rr_mv = raw_records_aux_mv

        extra_offset = 0
        _mask_tpc = (rr_tpc['channel'] == AqmonChannels.GPS_SYNC)
        if not np.any(_mask_tpc):
            # For some runs in the beginning no signal has been acquired here.
            # In that case we have to add the internal DAQ delay as an extra offset later.
            _mask_tpc = (rr_tpc['channel'] == AqmonChannels.GPS_SYNC_AM)
            extra_offset = self.get_delay()

        hits_tpc = self.get_nim_edge(rr_tpc[_mask_tpc], self.config['adc_threshold_nim_signal'])
        hits_tpc['time'] += extra_offset

        _mask_mveto = (rr_mv['channel'] == AqmonChannels.GPS_SYNC_MV)
        hits_mv = self.get_nim_edge(rr_mv[_mask_mveto], self.config['adc_threshold_nim_signal'])

        _mask_nveto = rr_nv['channel'] == AqmonChannels.GPS_SYNC_NV
        hits_nv = self.get_nim_edge(rr_nv[_mask_nveto], self.config['adc_threshold_nim_signal'])
        nveto_extra_offset = 0
        if not len(hits_nv):
            # During SR0 sync signal was not recorded properly for the
            # neutron-veto, hence take waveform itself as "hits".
            _mask_nveto &= rr_nv['record_i'] == 0
            nveto_extra_offset = self.config['epsilon_offset']
            hits_nv = rr_nv[_mask_nveto]
        hits_nv['time'] += nveto_extra_offset

        offsets_mv = self.estimate_delay(hits_tpc, hits_mv)
        offsets_nv = self.estimate_delay(hits_tpc, hits_nv)
        assert len(offsets_mv) == len(offsets_nv), 'Unequal number of sync signals!'

        result = np.zeros(len(offsets_mv), dtype=self.dtype)
        result['time'] = hits_tpc['time']
        result['endtime'] = strax.endtime(hits_tpc)
        result['time_offset_nv'] = offsets_nv
        result['time_offset_mv'] = offsets_mv

        return result

    def get_delay(self):
        delay = 0
        for run_id, _delay in self.config['tpc_internal_delay'].items():
            if int(self.run_id) >= int(run_id):
                delay = _delay
        return delay

    @staticmethod
    def get_nim_edge(raw_records, threshold=500):
        records = strax.raw_to_records(raw_records)
        strax.baseline(records)
        hits = strax.find_hits(records, min_amplitude=threshold)
        return hits

    def estimate_delay(self, hits_det0, hits_det1):
        """
        Function to estimate the average offset between two hits.
        """
        err_value = -10000000000

        offsets = []
        prev_time = 0
        for ind in range(len(hits_det0)):
            offset = self.find_offset_nearest(hits_det1['time'], hits_det0['time'][ind])
            if ind:
                # Cannot compute time to prev for first event
                time_to_prev = hits_det0['time'][ind] - prev_time
            else:
                time_to_prev = 10e9

            # Additional check to avoid spurious signals
            _correct_distance_to_prev_lock = time_to_prev >= self.sync_expected_min_clock_distance
            _correct_distance_to_prev_lock = time_to_prev < self.sync_expected_max_clock_distance
            if (abs(offset) < self.sync_max_delay) & _correct_distance_to_prev_lock:
                offsets.append(offset)
                prev_time = hits_det0['time'][ind]
            else:
                # Add err_value in case offset is not valid
                offsets.append(err_value)
                prev_time = hits_det0['time'][ind]

        return np.array(offsets)

    def find_offset_nearest(self, array, value):
        if not len(array):
            return -self.sync_max_delay
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return value-array[idx]
