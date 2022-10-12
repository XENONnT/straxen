import strax
import straxen

import numpy as np
import numba
import pandas as pd

import typing as ty
from immutabledict import immutabledict

export, __all__ = strax.exporter()


class nVETOEvents(strax.OverlapWindowPlugin):
    """
    Plugin which computes the boundaries of veto events.
    """
    __version__ = '0.0.3'

    depends_on = 'hitlets_nv'
    provides = 'events_nv'
    data_kind = 'events_nv'
    compressor = 'zstd'
    events_seen = 0

    event_left_extension_nv = straxen.URLConfig(
        default=0,
        track=True,
        type=int,
        help='Extends event window this many [ns] to the left.'
    )

    event_resolving_time_nv = straxen.URLConfig(
        default=200,
        track=True,
        type=int,
        help='Resolving time for window coincidence [ns].'
    )

    event_min_hits_nv = straxen.URLConfig(
        default=3,
        track=True,
        type=int,
        help='Minimum number of fully confined hitlets to define an event.'
    )

    channel_map = straxen.URLConfig(
        track=False,
        type=immutabledict,
        help='immutabledict mapping subdetector to (min, max) channel number'
    )

    def infer_dtype(self):
        self.name_event_number = 'event_number_nv'
        self.channel_range = self.channel_map['nveto']
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1
        return veto_event_dtype(self.name_event_number, self.n_channel)

    def get_window_size(self):
        return self.event_left_extension_nv + self.event_resolving_time_nv + 1

    def compute(self, hitlets_nv, start, end):

        events, hitlets_ids_in_event = find_veto_events(hitlets_nv,
                                                        self.event_min_hits_nv,
                                                        self.event_resolving_time_nv,
                                                        self.event_left_extension_nv,
                                                        event_number_key=self.name_event_number,
                                                        n_channel=self.n_channel,)

        if len(hitlets_ids_in_event):
            compute_nveto_event_properties(events,
                                           hitlets_nv,
                                           hitlets_ids_in_event,
                                           start_channel=self.channel_range[0])

        # Get eventids:
        n_events = len(events)
        events[self.name_event_number] = np.arange(n_events) + self.events_seen
        self.events_seen += n_events

        # Don't extend beyond the chunk boundaries
        # This will often happen for events near the invalid boundary of the
        # overlap processing (which should be thrown away)
        events['time'] = np.clip(events['time'], start, end)
        events['endtime'] = np.clip(events['endtime'], start, end)
        return events


def veto_event_dtype(name_event_number: str = 'event_number_nv',
                     n_pmts: int = 120) -> list:
    dtype = []
    dtype += strax.time_fields  # because mutable
    dtype += [(('Veto event number in this dataset', name_event_number), np.int64),
              (('Total area of all hitlets in event [pe]', 'area'), np.float32),
              (('Total number of hitlets in events', 'n_hits'), np.int32),
              (('Total number of contributing channels', 'n_contributing_pmt'), np.uint8),
              (('Area in event per channel [pe]', 'area_per_channel'), np.float32, n_pmts),
              (('Area weighted mean time of the event relative to the event start [ns]',
                'center_time'), np.float32),
              (('Weighted variance of time [ns]', 'center_time_spread'), np.float32),
              ]
    return dtype


@numba.njit(cache=True, nogil=True)
def compute_nveto_event_properties(events: np.ndarray,
                                   hitlets: np.ndarray,
                                   contained_hitlets_ids: np.ndarray,
                                   start_channel: int = 2000):
    """
    Computes properties of the neutron-veto events. Writes results
    directly to events.

    :param events: Events for which properties should be computed
    :param hitlets: hitlets which were used to build the events.
    :param contained_hitlets_ids: numpy array of the shape n x 2 which holds
        the indices of the hitlets contained in the corresponding event.
    :param start_channel: Integer specifying start channel, e.g. 2000
        for nveto.
    """
    for e, (s_i, e_i) in zip(events, contained_hitlets_ids):
        hitlet = hitlets[s_i:e_i]
        event_area = np.sum(hitlet['area'])
        e['area'] = event_area
        e['n_hits'] = len(hitlet)
        e['n_contributing_pmt'] = len(np.unique(hitlet['channel']))

        t = hitlet['time'] - hitlet[0]['time']
        if event_area:
            e['center_time'] = np.sum(t * hitlet['area']) / event_area
            if e['n_hits'] > 1 and e['center_time']:
                w = hitlet['area'] / e['area']  # normalized weights
                # Definition of variance
                e['center_time_spread'] = np.sqrt(np.sum(w * np.power(t - e['center_time'], 2)) / np.sum(w))
            else:
                e['center_time_spread'] = np.inf

        # Compute per channel properties:
        for hit in hitlet:
            ch = hit['channel'] - start_channel
            e['area_per_channel'][ch] += hit['area']


@export
def find_veto_events(hitlets: np.ndarray,
                     coincidence_level: int,
                     resolving_time: int,
                     left_extension: int,
                     event_number_key: str = 'event_number_nv',
                     n_channel: int = 120, ) -> ty.Tuple[np.ndarray, np.ndarray]:
    """
    Function which find the veto events as a nfold concidence in a given
    resolving time window. All hitlets which touch the event window
    contribute.

    :param hitlets: Hitlets which shall be used for event creation.
    :param coincidence_level: int, coincidence level.
    :param resolving_time: int, resolving window for coincidence in ns.
    :param left_extension: int, left event extension in ns.
    :param event_number_key: str, field name for the event number
    :param n_channel: int, number of channels in detector.
    :returns: events, hitelt_ids_per_event
    """
    # Find intervals which satisfy requirement:
    event_intervals = straxen.plugins.nveto_recorder.find_coincidence(hitlets,
                                                                      coincidence_level,
                                                                      resolving_time,
                                                                      left_extension,)

    # Find all hitlets which touch the coincidence windows:
    # (we cannot use fully_contained in here since some muon signals
    # may be larger than 300 ns)
    hitlets_ids_in_event = strax.touching_windows(hitlets,
                                                  event_intervals)

    # For some rare cases long signals may touch two intervals, in that
    # case we merge the intervals in the subsequent function:
    hitlets_ids_in_event = _solve_ambiguity(hitlets_ids_in_event)

    # Now we can create the veto events:
    events = np.zeros(len(hitlets_ids_in_event),
                      dtype=veto_event_dtype(event_number_key, n_channel))
    _make_event(hitlets, hitlets_ids_in_event, events)
    return events, hitlets_ids_in_event


@numba.njit(cache=True, nogil=False)
def _solve_ambiguity(contained_hitlets_ids: np.ndarray) -> np.ndarray:
    """
    Function which solves the ambiguity if a single hitlets overlaps
    with two event intervals.

    This can happen for muon signals which have a long tail, since we
    define the coincidence window as a fixed window. Hence those tails
    can extend beyond the fixed window.
    """
    res = np.zeros(contained_hitlets_ids.shape, dtype=contained_hitlets_ids.dtype)

    if not len(res):
        # Return empty result
        return res

    offset = 0
    start_i, end_i = contained_hitlets_ids[0]
    for e_i, ids in enumerate(contained_hitlets_ids[1:]):

        if end_i > ids[0]:
            # Current and next interval overlap so just updated the end
            # index.
            end_i = ids[1]
        else:
            # They do not overlap store indices:
            res[offset] = [start_i, end_i]
            offset += 1
            # Init next interval:
            start_i, end_i = ids

    # Last event:
    res[offset, :] = [start_i, end_i]
    offset += 1
    return res[:offset]


@numba.njit(cache=True, nogil=True)
def _make_event(hitlets: np.ndarray,
                hitlet_ids: np.ndarray,
                res: np.ndarray):
    """
    Function which sets veto event time and endtime.
    """
    for ei, ids in enumerate(hitlet_ids):
        hit = hitlets[ids[0]:ids[1]]
        res[ei]['time'] = hit[0]['time']
        res[ei]['endtime'] = np.max(strax.endtime(hit))


class nVETOEventPositions(strax.Plugin):
    """
    Plugin which computes the interaction position in the nveto as an
    azimuthal angle.
    """
    __version__ = '0.1.1'

    depends_on = ('events_nv', 'hitlets_nv')
    data_kind = 'events_nv'
    provides = 'event_positions_nv'
    compressor = 'zstd'

    position_max_time_nv = straxen.URLConfig(default=20, infer_type=False,
                 help="Time [ns] within an event use to compute the azimuthal angle of the "
                      "event.")

    nveto_pmt_position_map = straxen.URLConfig(
                 help="nVeto PMT position mapfile",
                 default='resource://nveto_pmt_position.csv?fmt=csv', infer_type=False,)

    def infer_dtype(self):
        return veto_event_positions_dtype()

    def setup(self):
        npmt_pos = self.nveto_pmt_position_map
        # Use records instead of a dataframe.
        self.pmt_properties = npmt_pos.to_records(index=False)

    def compute(self, events_nv, hitlets_nv):
        event_angles = np.zeros(len(events_nv), dtype=self.dtype)

        # Split hitlets by containment, works since we updated event start/end in
        # compute_event_properties.
        hits_in_events = strax.split_by_containment(hitlets_nv, events_nv)

        # Compute hitlets within the first x ns of event:
        hits_in_events, n_prompt = first_hitlets(hits_in_events,
                                                 self.position_max_time_nv)
        event_angles['n_prompt_hitlets'] = n_prompt

        # Compute azimuthal angle and xyz positions:
        angle = get_average_angle(hits_in_events,
                                  self.pmt_properties)
        event_angles['angle'] = angle
        compute_positions(event_angles, hits_in_events, self.pmt_properties)
        strax.copy_to_buffer(events_nv, event_angles, f'_copy_events_nv')

        return event_angles


def veto_event_positions_dtype() -> list:
    dtype = []
    dtype += strax.time_fields
    dtype += [(('Number of prompt hitlets within the first "position_max_time_nv" ns of the event.',
                'n_prompt_hitlets'), np.int16),
              (('Azimuthal angle, where the neutron capture was detected in [0, 2 pi).',
                'angle'), np.float32),
              (('Area weighted mean of position in x [mm]', 'pos_x'), np.float32),
              (('Area weighted mean of position in y [mm]', 'pos_y'), np.float32),
              (('Area weighted mean of position in z [mm]', 'pos_z'), np.float32),
              (('Weighted variance of position in x [mm]', 'pos_x_spread'), np.float32),
              (('Weighted variance of position in y [mm]', 'pos_y_spread'), np.float32),
              (('Weighted variance of position in z [mm]', 'pos_z_spread'), np.float32)
              ]
    return dtype


@numba.njit(cache=True, nogil=True)
def compute_positions(event_angles: np.ndarray,
                      contained_hitlets: numba.typed.typedlist.List,
                      pmt_pos: np.ndarray,
                      start_channel: int = 2000):
    """
    Function which computes some artificial event position for a given
    neutron/muon-veto event. The position is computed based on a simple
    area weighted mean. Please note that the event position can be
    reconstructed in unphysical regions like being within the TPC.

    :param event_angles: Result array of the veto_event_position dtype.
        The result is updated inplace.
    :param contained_hitlets: Hitlets contained in each event.
    :param pmt_pos: Position of the veto PMTs
    :param start_channel: Starting channel of the detector.
    """
    for e_angles, hitlets in zip(event_angles, contained_hitlets):
        prompt_event_area = np.sum(hitlets['area'])
        if prompt_event_area:
            ch = hitlets['channel'] - start_channel
            pos_x = pmt_pos['x'][ch]
            pos_y = pmt_pos['y'][ch]
            pos_z = pmt_pos['z'][ch]

            e_angles['pos_x'] = np.sum(pos_x * hitlets['area'])/prompt_event_area
            e_angles['pos_y'] = np.sum(pos_y * hitlets['area'])/prompt_event_area
            e_angles['pos_z'] = np.sum(pos_z * hitlets['area'])/prompt_event_area
            w = hitlets['area'] / prompt_event_area  # normalized weights
            if len(hitlets) and np.sum(w) > 0:
                e_angles['pos_x_spread'] = np.sqrt(
                    np.sum(w * (pos_x - e_angles['pos_x'])**2)/np.sum(w)
                )
                e_angles['pos_y_spread'] = np.sqrt(
                    np.sum(w * (pos_y - e_angles['pos_y'])**2)/np.sum(w)
                )
                e_angles['pos_z_spread'] = np.sqrt(
                    np.sum(w * (pos_z - e_angles['pos_z'])**2)/np.sum(w)
                )


@numba.njit(cache=True, nogil=True)
def get_average_angle(hitlets_in_event: numba.typed.typedlist.List,
                      pmt_properties: np.ndarray,
                      start_channel: int = 2000, ) -> np.ndarray:
    """
    Computes azimuthal angle as an area weighted mean over all hitlets.

    :param hitlets_in_event: numba.typed.List containing the hitlets per
        event.
    :param pmt_properties: numpy.sturctured.array containing the PMT
        positions in the fields "x" and "y".
    :param start_channel: First channel e.g. 2000 for nevto.
    :return: np.array holding the azimuthal angles.
    """
    res = np.zeros(len(hitlets_in_event), np.float32)
    for ind, hitlets in enumerate(hitlets_in_event):
        if np.sum(hitlets['area']):
            x = pmt_properties['x'][hitlets['channel'] - start_channel]
            y = pmt_properties['y'][hitlets['channel'] - start_channel]

            weighted_mean_x = np.sum(x * hitlets['area']) / np.sum(hitlets['area'])
            weighted_mean_y = np.sum(y * hitlets['area']) / np.sum(hitlets['area'])
            res[ind] = _circ_angle(weighted_mean_x, weighted_mean_y)
        else:
            res[ind] = np.nan
    return res


@numba.njit(cache=True, nogil=True)
def circ_angle(x_values: np.ndarray,
               y_values: np.ndarray) -> np.ndarray:
    """
    Loops over a set of x and y values and computes azimuthal angle.

    :param x_values: x-coordinates
    :param y_values: y-coordinates
    :return: angles
    """
    res = np.zeros(len(x_values), dtype=np.float32)
    for ind, (x, y) in enumerate(zip(x_values, y_values)):
        res[ind] = _circ_angle(x, y)
    return res


@numba.njit(cache=True, nogil=True)
def _circ_angle(x: float, y: float) -> float:
    if x > 0 and y >= 0:
        # 1st quadrant
        angle = np.abs(np.arctan(y / x))
        return angle
    elif x <= 0 and y > 0:
        # 2nd quadrant
        angle = np.abs(np.arctan(x / y))
        return np.pi / 2 + angle
    elif x < 0 and y <= 0:
        # 3rd quadrant
        angle = np.abs(np.arctan(y / x))
        return np.pi + angle
    elif y < 0 and x >= 0:
        # 4th quadrant
        angle = np.abs(np.arctan(x / y))
        return 3 / 2 * np.pi + angle
    elif x == 0 and y == 0:
        return np.NaN
    else:
        print(x, y)
        raise ValueError('It should be impossible to arrive here, '
                         'but somehow we managed.')


@numba.njit(cache=True, nogil=True)
def first_hitlets(hitlets_per_event: np.ndarray,
                  max_time: int) -> ty.Tuple[numba.typed.List, np.ndarray]:
    """
    Returns hitlets within the first "max_time" ns of an event.

    :param hitlets_per_event: numba.typed.List of hitlets per event.
    :param max_time: int max allowed time difference to leading hitlet
        in ns.
    """
    res_hitlets_in_event = numba.typed.List()
    res_n_prompt = np.zeros(len(hitlets_per_event), np.int16)
    for ind, hitlets in enumerate(hitlets_per_event):
        m = (hitlets['time'] - hitlets[0]['time']) < max_time
        h = hitlets[m]
        res_hitlets_in_event.append(h)
        res_n_prompt[ind] = len(h)
    return res_hitlets_in_event, res_n_prompt


class muVETOEvents(nVETOEvents):
    """Plugin which computes the boundaries of veto events.
    """
    depends_on = 'hitlets_mv'
    provides = 'events_mv'
    data_kind = 'events_mv'

    compressor = 'zstd'
    child_plugin = True

    __version__ = '0.0.1'
    events_seen = 0

    event_left_extension_mv = straxen.URLConfig(
        default=0,
        track=True,
        type=int,
        child_option=True,
        parent_option_name='event_left_extension_nv',
        help='Extends event window this many [ns] to the left.'
    )
    event_resolving_time_mv = straxen.URLConfig(
        default=300,
        track=True,
        type=int,
        child_option=True,
        parent_option_name='event_resolving_time_nv',
        help='Resolving time for window coincidence [ns].'
    )
    event_min_hits_mv = straxen.URLConfig(
        default=3,
        track=True,
        type=int,
        child_option=True,
        parent_option_name='event_min_hits_nv',
        help='Minimum number of fully confined hitlets to define an event.'
    )

    def infer_dtype(self):
        self.name_event_number = 'event_number_mv'
        self.channel_range = self.channel_map['mv']
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1
        return veto_event_dtype(self.name_event_number, self.n_channel)

    def get_window_size(self):
        return self.event_left_extension_mv + self.event_resolving_time_mv + 1

    def compute(self, hitlets_mv, start, end):
        return super().compute(hitlets_mv, start, end)


class nVETOEventsSync(strax.OverlapWindowPlugin):
    """
    Plugin which computes time stamps which are synchronized with the
    TPC. Uses delay set in the DAQ.
    """
    depends_on = ('events_nv', 'detector_time_offsets')
    delay_field_name = 'time_offset_nv'

    provides = 'events_sync_nv'
    save_when = strax.SaveWhen.EXPLICIT
    __version__ = '0.0.3'

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields
        dtype += [(('Time of the event synchronized according to the total digitizer delay.',
                    'time_sync'), np.int64),
                  (('Endtime of the event synchronized according to the total digitizer delay.',
                    'endtime_sync'), np.int64),
                  ]
        return dtype

    def get_window_size(self):
        # Ensure to have at least 12 offset-values from detector_time_offsets
        # to compute average time delay. Otherwise we may get unlucky with
        # our pacemaker (unlikely but could happen).
        return 120*10**9

    def compute(self, events_nv, detector_time_offsets):
        delay = detector_time_offsets[self.delay_field_name]
        delay = np.median(delay[delay > 0])
        delay = delay.astype(np.int64)
        # Check if delay is >= 0 otherwise something went wrong with 
        # the sync signal.
        assert delay >= 0, f'Missing the GPS sync signal for run {self.run_id}.' 
        
        events_sync_nv = np.zeros(len(events_nv), self.dtype)
        events_sync_nv['time'] = events_nv['time']
        events_sync_nv['endtime'] = events_nv['endtime']
        events_sync_nv['time_sync'] = events_nv['time'] + delay
        events_sync_nv['endtime_sync'] = events_nv['endtime'] + delay
        return events_sync_nv


class mVETOEventSync(nVETOEventsSync):
    """
    Plugin which computes synchronized timestamps for the muon-veto with
    respect to the TPC.
    """
    depends_on = ('events_mv', 'detector_time_offsets')
    delay_field_name = 'time_offset_mv'

    provides = 'events_sync_mv'
    __version__ = '0.0.1'
    child_plugin = True

    def compute(self, events_mv, detector_time_offsets):
        return super().compute(events_mv, detector_time_offsets)
