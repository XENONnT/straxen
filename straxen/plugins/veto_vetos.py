import strax
import straxen

import numpy as np
import numba

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        'min_veto_area_nv', default=10, type=float, track=True,
        help='Minimal area required in pe to trigger veto.'),
    strax.Option(
        'min_veto_hits_nv', default=10, type=int, track=True,
        help='Minimal number of hitlets in event to trigger veto.'),
    strax.Option(
        'min_veto_channel_nv', default=0, type=int, track=True,
        help='Minimal number PMT channel contributing to the event.'),
    strax.Option(
        'veto_left_nv', default=1_000_000, type=int, track=True,
        help='Veto time in ns left t the start of a vetoing event.'),
    strax.Option(
        'veto_right_nv', default=2_600_000, type=int, track=True,
        help='Veto time in ns right to the end of a vetoing event.'),
)
class nVETOveto(strax.OverlapWindowPlugin):
    """
    This is a default plugin, like used by many plugins in straxen. It
    takes hitlets and multipliers their area with a user defined
    multiplier.
    """
    __version__ = '0.0.1'

    depends_on = 'events_nv'
    provides = 'veto_nv'
    data_kind = 'veto_nv'

    dtype = strax.time_fields

    def get_window_size(self):
        # Take a large window for safety, events can be ~500 ns large
        return 10 * (self.config['veto_left_nv']
                     + self.config['veto_right_nv'])

    def compute(self, events_nv, start, end):
        vetos = create_veto_intervals(events_nv,
                                      self.config['min_veto_area_nv'],
                                      self.config['min_veto_hits_nv'],
                                      self.config['veto_left_nv'],
                                      self.config['veto_right_nv'])

        # Now we have to do clip all times and end endtimes in case
        # they go beyond a chunk boundary:
        vetos['time'] = np.clip(vetos['time'], start, end)
        vetos['endtime'] = np.clip(vetos['endtime'], start, end)

        return vetos


def create_veto_intervals(events,
                          min_area,
                          min_hits,
                          min_contributing_channels,
                          left_extension,
                          right_extension, ):
    """
    Function which creates veto regions.

    :param events: nveto events
    :param min_area: min area in pe required to create a veto region.
    :param min_hits: same but with hitlets
    :param min_contributing_channels: Minimal number of contributing
        channels.
    :param left_extension: Left extension of the event to define veto
        region in ns.
    :param right_extension: Same but right hand side after.

    :returns: numpy.structured.array containing the veto regions.
    """
    res = np.zeros(len(events),
                   dtype=strax.time_fields)

    res = _create_veto_intervals(events,
                                 min_area,
                                 min_hits,
                                 min_contributing_channels,
                                 left_extension,
                                 right_extension,
                                 res, )
    res = merge_intervals(res)

    return res


@numba.njit(cache=True)
def _create_veto_intervals(events,
                           min_area,
                           min_hits,
                           min_contributing_channels,
                           left_extension,
                           right_extension,
                           res, ):
    offset = 0

    for ev in events:
        satisfies_veto_trigger = (ev['area'] >= min_area
                                  or ev['n_hits'] >= min_hits
                                  or ev['n_contributing_pmt'] >= min_contributing_channels)
        if not satisfies_veto_trigger:
            continue

        res[offset]['time'] = ev['time'] - left_extension
        res[offset]['endtime'] = ev['endtime'] + right_extension
        offset += 1
    return res[:offset]


def merge_intervals(intervals):
    """
    Function which merges overlapping intervals into a single one.
    """
    res = np.zeros(len(intervals), dtype=strax.time_fields)
    res = _merge_intervals(intervals['time'],
                           intervals['endtime'],
                           res)
    return res


@numba.njit(cache=True, nogil=True)
def _merge_intervals(start, end, res):
    offset = 0

    int_s = start[0]
    int_e = end[0]
    for s, e in zip(start[1:], end[1:]):
        if int_e >= s:
            # Interval overlaps, updated only end:
            int_e = e
            continue

        # Intervals do not overlap, save interval:
        res[offset]['time'] = int_s
        res[offset]['endtime'] = int_e
        offset += 1

        # New interval:
        int_s = s
        int_e = e

    # Save last interval:
    res[offset]['time'] = int_s
    res[offset]['endtime'] = int_e
    offset += 1

    return res[:offset]
