import strax
import straxen
import numpy as np
import numba

export, __all__ = strax.exporter()
MV_PREAMBLE = 'Muno-Veto Plugin: Same as the corresponding nVETO-Plugin.\n'


@export
@strax.takes_config(
    strax.Option(
        'min_veto_area_nv', default=5, type=float, track=True,
        help='Minimal area required in pe to trigger veto.'),
    strax.Option(
        'min_veto_hits_nv', default=10, type=int, track=True,
        help='Minimal number of hitlets in n/mveto_event to trigger a veto.'),
    strax.Option(
        'min_veto_channel_nv', default=0, type=int, track=True,
        help='Minimal number PMT channel contributing to the n/mveto_event.'),
    strax.Option(
        'veto_left_extension_nv', default=500_000, type=int, track=True,
        help='Veto time in ns left t the start of a vetoing event.'),
    strax.Option(
        'veto_right_extension_nv', default=0, type=int, track=True,
        help='Veto time in ns right to the end of a vetoing event.'),
)
class nVETOVetoRegions(strax.OverlapWindowPlugin):
    """
    Plugin which defines the time intervals in which peaks should be
    tagged as vetoed. An event must surpass all three criteria to trigger
    a veto.
    """
    __version__ = '0.0.1'

    depends_on = ('events_nv', 'events_sync_nv')
    provides = 'veto_regions_nv'
    data_kind = 'veto_regions_nv'
    save_when = strax.SaveWhen.NEVER

    dtype = strax.time_fields

    def get_window_size(self):
        return 10 * (self.config['veto_left_extension_nv'] + self.config['veto_right_extension_nv'])

    def compute(self, events_nv, start, end):
        vetos = create_veto_intervals(events_nv,
                                      self.config['min_veto_area_nv'],
                                      self.config['min_veto_hits_nv'],
                                      self.config['min_veto_channel_nv'],
                                      self.config['veto_left_extension_nv'],
                                      self.config['veto_right_extension_nv'])

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
    res = straxen.merge_intervals(res)

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
                                  and ev['n_hits'] >= min_hits
                                  and ev['n_contributing_pmt'] >= min_contributing_channels)
        if not satisfies_veto_trigger:
            continue

        res[offset]['time'] = ev['time_sync'] - left_extension
        res[offset]['endtime'] = ev['endtime_sync'] + right_extension
        offset += 1
    return res[:offset]


@strax.takes_config(
    strax.Option(
        'min_veto_area_mv', default=10, type=float, track=True,
        child_option=True, parent_option_name='min_veto_area_nv',
        help='Minimal area required in pe to trigger veto.'),
    strax.Option(
        'min_veto_hits_mv', default=0, type=int, track=True,
        child_option=True, parent_option_name='min_veto_hits_nv',
        help='Minimal number of hitlets in event to trigger veto.'),
    strax.Option(
        'min_veto_channel_mv', default=5, type=int, track=True,
        child_option=True, parent_option_name='min_veto_channel_nv',
        help='Minimal number PMT channel contributing to the event.'),
    strax.Option(
        'veto_left_extension_mv', default=0, type=int, track=True,
        child_option=True, parent_option_name='veto_left_extension_nv',
        help='Veto time in ns left t the start of a vetoing event.'),
    strax.Option(
        'veto_right_extension_mv', default=1_000_000, type=int, track=True,
        child_option=True, parent_option_name='veto_right_extension_nv',
        help='Veto time in ns right to the end of a vetoing event.'),
)
class muVETOVetoRegions(nVETOVetoRegions):
    __doc__ = MV_PREAMBLE + nVETOVetoRegions.__doc__
    __version__ = '0.0.1'

    depends_on = ('events_mv', 'events_sync_mv')
    provides = 'veto_regions_mv'
    data_kind = 'veto_regions_mv'
    save_when = strax.SaveWhen.NEVER

    dtype = strax.time_fields
    child_plugin = True

    def get_window_size(self):
        return 10 * (self.config['veto_left_extension_mv'] + self.config['veto_left_extension_mv'])

    def compute(self, events_mv, start, end):
        return super().compute(events_mv, start, end)
