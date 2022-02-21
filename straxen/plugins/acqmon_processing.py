import typing
import warnings
from enum import IntEnum
import numba
import numpy as np
import strax
import straxen

from .daqreader import ARTIFICIAL_DEADTIME_CHANNEL

export, __all__ = strax.exporter()


# More info about the acquisition monitor can be found here:
# https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:alexelykov:acquisition_monitor

class AqmonChannels(IntEnum):
    """Mapper of named aqmon channels to ints"""
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


@export
class AqmonHits(strax.Plugin):
    """
    Find hits in acquisition monitor data. These hits could be
    then used by other plugins for deadtime calculations,
    GPS SYNC analysis, etc.
    """
    save_when = strax.SaveWhen.TARGET
    __version__ = '1.0.1'
    hit_min_amplitude_aqmon = straxen.URLConfig(
        default=(
            # Analogue signals
            (50, (int(AqmonChannels.SUM_WF),)),
            # Digital signals, can set a much higher threshold
            (1500, (
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

    depends_on = 'raw_records_aqmon'
    provides = 'aqmon_hits'
    data_kind = 'aqmon_hits'

    dtype = strax.hit_dtype

    def compute(self, raw_records_aqmon):
        not_allowed_channels = (set(np.unique(raw_records_aqmon['channel']))
                                - set(self.aqmon_channels))
        if not_allowed_channels:
            raise ValueError(
                f'Unknown channel {not_allowed_channels}. Only know {self.aqmon_channels}')
        records = strax.raw_to_records(raw_records_aqmon)
        strax.sort_by_time(records)
        strax.zero_out_of_bounds(records)
        strax.baseline(records, baseline_samples=self.baseline_samples_aqmon, flip=True)
        aqmon_hits = self.find_aqmon_hits_per_channel(records)
        return aqmon_hits

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
            strax.sort_by_time(aqmon_hits)
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
        busy_*  <= V1495 busy veto for tpc channels
        busy_he_*    <= V1495 busy veto for high energy tpc channels
        hev_*   <= DDC10 hardware high energy veto
        straxen_deadtime <= special case of deadtime introduced by the
            DAQReader-plugin
    """
    __version__ = '1.0.1'
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

    def send_input_after(self):
        """After how long buffering can we start sending the output"""
        return 10e9

    def min_gap_in_chunk(self):
        """How long does the gap need to be minimally to the next chunk"""
        return 10e9

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
                and len(veto_hits_start)
                and veto_hits_start[-1]['time'] > veto_hits_stop['time'][-1])

        if missing_a_final_stop:
            # There is one *start* of the //end// of the run -> the
            # **stop** is missing (because it's outside of the run),
            # let's add one **stop** at the //end// of this chunk
            warnings.warn(f'Inserted one end for {self.run_id} at {chunk_end}')
            extra_stop = self.fake_hit(chunk_end)
            veto_hits_stop = np.concatenate([veto_hits_stop, extra_stop])
        if len(veto_hits_stop) - len(veto_hits_start) == 1:
            # There is one *stop* of the //beginning// of the run
            # -> the **start** is missing (because it's from before
            # starting the run), # let's add one **start** at the
            # //beginning// of this chunk
            warnings.warn(f'Inserted one start for {self.run_id} at {chunk_start}')
            extra_start = self.fake_hit(chunk_start)
            veto_hits_start = np.concatenate([extra_start, veto_hits_start])

        something_is_wrong = len(veto_hits_start) != len(veto_hits_stop)
        if something_is_wrong:
            message = (f'Got inconsistent number of {veto_name} '
                       f'starts ({len(veto_hits_start)}) / '
                       f'stops ({len(veto_hits_stop)}).')
            if len(extra_start):
                message += (' Despite the fact that we inserted one extra '
                            'start at the beginning of the run.')
            elif len(extra_stop):
                message += (' Despite the fact that we inserted one extra '
                            'stop at the end of the run.')
            if len(extra_start) and len(extra_stop):
                raise RuntimeError('Someone broke my code, this cannot happen?!')
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

    __version__ = '2.0.0'
    # Strictly speaking, we could depend on 'events', but then you couldn't
    # change the event_window_fields to e.g. s1_time and s2_endtime.
    depends_on = ('event_basics', 'veto_intervals')
    provides = 'veto_proximity'
    data_kind = 'events'
    save_when = strax.SaveWhen.NEVER

    event_window_fields = straxen.URLConfig(
        default=('time', 'endtime'),
        help='Fields to determine where to look for overlaps for using '
             'this plugin in the events. The default uses start and endtime '
             'of an event, but this can also be the S1 or S2 start/endtime'
    )

    veto_proximity_window = straxen.URLConfig(
        default=int(120e9),
        help='Maximum separation between veto stop and start pulses [ns]'
    )
    time_no_aqmon_veto_found = straxen.URLConfig(
        default=int(3.6e+12),
        track=True,
        type=int,
        help='If we don\'t find a veto close to the event, say that '
             'the closest event is this many ns removed from it.'
    )

    veto_names = ['busy', 'he', 'hev', 'straxen_deadtime']

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields
        start_field, stop_field = self.event_window_fields
        for name in self.veto_names:
            dtype += [
                ((f'Duration of event overlapping with "{name}"-veto [ns]',
                  f'veto_{name}_overlap'),
                 np.int64),
                ((f'Time (absolute) to previous "{name}"-veto '
                  f'from "{start_field}" of event [ns]',
                  f'time_to_previous_{name}'),
                 np.int64),
                ((f'Time (absolute) to next "{name}"-veto '
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
