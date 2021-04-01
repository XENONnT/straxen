import strax
import straxen

import numpy as np
import numba
from immutabledict import immutabledict


@strax.takes_config(
    strax.Option('event_left_extension_nv', default=0,
                 help="Extends events this many ns to the left"),
    strax.Option('event_resolving_time_nv', default=300,
                 help="Resolving time for fixed window coincidence [ns]."),
    strax.Option('event_min_hits_nv', default=3,
                 help="Minimum number of fully confined hitlets to define an event."),
    strax.Option('gain_model_nv',
                 help='PMT gain model. Specify as (model_type, model_config)'),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number."),
)
class nVETOEvents(strax.OverlapWindowPlugin):
    """
    Plugin which computes the boundaries of veto events.
    """
    depends_on = 'hitlets_nv'
    provides = 'events_nv'
    data_kind = 'events_nv'

    compressor = 'zstd'

    # Needed in case we make again an muVETO child.
    ends_with = '_nv'

    __version__ = '0.0.1'
    events_seen = 0

    def infer_dtype(self):
        self.name_event_number = 'event_number_nv'
        self.channel_range = self.config['channel_map']['nveto']
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1
        return veto_event_dtype(self.name_event_number, self.n_channel)

    def setup(self):
        self.to_pe = straxen.get_to_pe(self.run_id,
                                       self.config['gain_model_nv'],
                                       self.n_channel)

    def get_window_size(self):
        return self.config['event_left_extension_nv'] + self.config['event_resolving_time_nv'] + 1

    def compute(self, hitlets_nv, start, end):
        intervals = straxen.plugins.nveto_recorder.coincidence(hitlets_nv,
                                                               self.config['event_min_hits_nv'],
                                                               self.config['event_resolving_time_nv'],
                                                               self.config['event_left_extension_nv']
                                                               )

        n_events = len(intervals)
        events = np.zeros(n_events,
                          dtype=veto_event_dtype(self.name_event_number,
                                                 self.n_channel)
                          )
        events['time'] = intervals[:, 0]
        events['endtime'] = intervals[:, 1]

        # Find all hilets which touch the coincidence windows
        # and fix ambiguities in case a single hitlet touches two intveral
        # (this can happen for muon signals with a long tail + afterpulses/trapped
        # light, in those cases we merge the two events into a single one)
        hitlets_ids_in_event = strax.touching_windows(hitlets_nv, events)
        hitlets_ids_in_event = solve_ambiguity(hitlets_ids_in_event)

        # Drop empty events (which were merged before):
        m = np.diff(hitlets_ids_in_event).flatten() != 0
        hitlets_ids_in_event = hitlets_ids_in_event[m]
        events = events[m]

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


def veto_event_dtype(name_event_number='event_number_nv', n_pmts=120):
    dtype = []
    dtype += strax.time_fields  # because mutable
    dtype += [(('Veto event number in this dataset', name_event_number), np.int64),
              (('Last hitlet endtime in event [ns].', 'last_hitlet_endtime'), np.int64),
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
def compute_nveto_event_properties(events, hitlets, contained_hitlets_ids, start_channel=2000):
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
        e['n_contributed_pmt'] = len(np.unique(hitlet['channel']))

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

        # Compute endtime of last hitlet in event:
        endtime = strax.endtime(hitlet)
        e['last_hitlet_endtime'] = np.max(endtime)

        # Update start and endtime as hitlets only have to overlap
        # partially
        e['time'] = min(e['time'], hitlet['time'][0])
        e['endtime'] = max(e['endtime'], max(endtime))


@numba.njit(cache=True, nogil=False)
def solve_ambiguity(contained_hitlets_ids):
    """
    Function which solves the ambiguity if a single hitlete overlaps
    with two event intervals.

    This can happen for muon signals which have a long tail, since we
    define the coincidence window as a fixed window. Hence those tails
    can extend beyond the fixed window.
    """
    res = np.zeros(contained_hitlets_ids.shape, dtype=contained_hitlets_ids.dtype)
    offset = 0
    skip_next = False
    for e_i, ids in enumerate(contained_hitlets_ids[:-1]):
        if skip_next:
            # Prev interval overlapped with current interval and was
            # merged so skip this one.
            skip_next = False
            continue

        # Test if current and next interval overlap:
        c_s_i, c_e_i = ids
        n_s_i, n_e_i = contained_hitlets_ids[e_i + 1]
        if (c_e_i - n_s_i) > 0:
            # Yes, they do so merge them increase the counter
            # by two to keep track of the events which were
            # merged.
            res[offset, :] = c_s_i, n_e_i
            offset += 2
            skip_next = True
        else:
            # No so just copy the values.
            res[offset, :] = c_s_i, c_e_i
            offset += 1

    # Last event:
    if not skip_next:
        res[offset, :] = contained_hitlets_ids[-1]
        offset += 1
    return res[:offset]


@strax.takes_config(
    strax.Option('angle_max_time_nv', default=10,
                 help="Time [ns] within an evnet use to compute the azimuthal angle of the event."),
    strax.Option('nveto_pmt_position_map', track=False,
                 help="nVeto PMT position mapfile",
                default='nveto_pmt_position.csv'),
)
class nVETOEventPositions(strax.Plugin):
    """
    Plugin which computes the interaction position in the nveto as an
    azimuthal angle.
    """
    depends_on = ('events_nv', 'hitlets_nv')
    data_kind = 'events_nv'
    provides = 'event_positions_nv'

    loop_over = 'events_nv'
    compressor = 'zstd'
    save_when = strax.SaveWhen.TARGET

    # Needed in case we make again an muVETO child.
    ends_with = '_nv'

    dtype = []
    dtype += strax.time_fields
    dtype += [(('Azimuthal angle, where the neutron capture was detected in [0, 2 pi).',
                'angle'), np.float32),
              (('Area weighted mean of position in x [mm]', 'pos_x'), np.float32),
              (('Area weighted mean of position in y [mm]', 'pos_y'), np.float32),
              (('Area weighted mean of position in z [mm]', 'pos_z'), np.float32),
              (('Weighted variance of position in x [mm]', 'pos_x_spread'), np.float32),
              (('Weighted variance of position in y [mm]', 'pos_y_spread'), np.float32),
              (('Weighted variance of position in z [mm]', 'pos_z_spread'), np.float32)
          ]

    __version__ = '0.0.2'

    def setup(self):
        self.pmt_properties = 1
        df_pmt_pos = straxen.get_resource(self.config['nveto_pmt_position_map'],fmt='csv')
        self.pmt_pos = df_pmt_pos.to_numpy(dtype=np.float32)

    def compute(self, events_nv, hitlets_nv):
        hits_in_events = strax.split_by_containment(hitlets_nv, events_nv)
        event_angles = np.zeros(len(events_nv), dtype=self.dtype)

        angle = compute_average_angle(hits_in_events,
                                      self.pmt_properties)

        event_angles['angle'] = angle
        compute_positions(event_angles, events_nv, hits_in_events, self.pmt_pos)
        strax.copy_to_buffer(events_nv, event_angles, '_copy_events_nv')

        return event_angles

@numba.njit
def compute_positions(event_angles, events, contained_hitlets, pmt_pos, start_channel=2000):
    for e_angles, e, hitlets in zip(event_angles, events, contained_hitlets):
        if e['area']:
            pmt_x = pmt_pos[hitlets['channel']-start_channel, 2] # 2 is index of x
            e_angles['pos_x'] = np.sum(pmt_x * hitlets['area']) / e['area']
            pmt_y = pmt_pos[hitlets['channel']-start_channel, 3] # 3 is index of y
            e_angles['pos_y'] = np.sum(pmt_y * hitlets['area']) / e['area']
            pmt_z = pmt_pos[hitlets['channel']-start_channel, 4] # 4 is index of z
            e_angles['pos_z'] = np.sum(pmt_z * hitlets['area']) / e['area']
            if e['n_hits'] > 1:
                w = hitlets['area']/e['area'] # normalized weights
                e_angles['pos_x_spread'] = np.sqrt(np.sum(w*np.power(pmt_x-e_angles['pos_x'],2))/np.sum(w))
                e_angles['pos_y_spread'] = np.sqrt(np.sum(w*np.power(pmt_y-e_angles['pos_y'],2))/np.sum(w))
                e_angles['pos_z_spread'] = np.sqrt(np.sum(w*np.power(pmt_z-e_angles['pos_z'],2))/np.sum(w))


@numba.njit(cache=True, nogil=True)
def compute_average_angle(hitlets_in_event,
                          pmt_properties,
                          start_channel=2000,
                          max_time=10):
    """
    Computes azimuthal angle as an area weighted mean over all hitlets
    which arrive within a certain time window.

    :param hitlets_in_event: numba.typed.List containing the hitlets per
        event.
    :param pmt_properties: numpy.sturctured.array containing the PMT
        positions in the fields "x" and "y".
    :param start_channel: First channel e.g. 2000 for nevto.
    :param max_time: Time within a hitlet must arrive in order to be
        used in the computation.
    :return: np.array holding the azimuthal angles.
    """
    res = np.zeros(len(hitlets_in_event), np.float32)
    for ind, hitlets in enumerate(hitlets_in_event):
        m = (hitlets['time'] - hitlets[0]['time']) < max_time
        h = hitlets[m]
        x = pmt_properties['x'][h['channel'] - start_channel]
        y = pmt_properties['y'][h['channel'] - start_channel]

        weighted_mean_x = np.sum(x * h['area']) / np.sum(h['area'])
        weighted_mean_y = np.sum(y * h['area']) / np.sum(h['area'])
        res[ind] = _circ_angle(weighted_mean_x, weighted_mean_y)
    return res


@numba.njit(cache=True, nogil=True)
def circ_angle(x_values, y_values):
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
def _circ_angle(x, y):
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
