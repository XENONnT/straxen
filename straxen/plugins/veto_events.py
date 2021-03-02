import strax
import straxen

import numpy as np
import numba
from immutabledict import immutabledict


@strax.takes_config(
    strax.Option('event_left_extension_nv', default=0,
                 help="Extends events this many ns to the left"),
    strax.Option('event_resolving_time_nv', default=300,
                 help="Resolving time for fixed window concidene [ns]."),
    strax.Option('event_min_hits_nv', default=6,
                 help="Minimum number of fully convinet hitlets to define an event."),
    strax.Option('gain_model_nv',
                 help='PMT gain model. Specify as (model_type, model_config)'),
    strax.Option('n_nveto_pmts', type=int,
                 help='Number of TPC PMTs'),
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

    parallel = 'process'
    compressor = 'zstd'

    __version__ = '0.0.1'
    event_seen = 0

    def setup(self):
        self.to_pe = straxen.get_to_pe(self.run_id,
                                       self.config['gain_model_nv'],
                                       self.config['n_nveto_pmts'])

        self.channel_range = self.config['channel_map']['nveto']

    def infer_dtype(self):
        return veto_event_dtype()

    def get_window_size(self):
        return self.config['event_left_extension_nv'] + self.config['event_resolving_time_nv'] + 1

    def compute(self, hitlets_nv, start, end):
        intervals = straxen.plugins.nveto_recorder.coincidence(hitlets_nv,
                                                               self.config['event_min_hits_nv'],
                                                               self.config['event_resolving_time_nv'],
                                                               self.config['event_left_extension_nv']
                                                               )

        n_events = len(intervals)
        events = np.zeros(n_events, dtype=veto_event_dtype())

        # Don't extend beyond the chunk boundaries
        # This will often happen for events near the invalid boundary of the
        # overlap processing (which should be thrown away)
        events['time'] = np.clip(intervals[:, 0], start, end)
        events['endtime'] = np.clip(intervals[:, 1], start, end)

        # Compute center time:
        split_hitlets = strax.split_by_containment(hitlets_nv, events)
        compute_event_properties(events, split_hitlets, start_channel=2000)

        # Cut all those events for which we have less than self.config['event_min_hits_nv'] 
        # hitlets. (straxen.plugins.nveto_recorder.coincidence works with partially overlaping things)
        events = events[events['n_hits'] >= self.config['event_min_hits_nv']]
        n_events = len(events)
        events['event_number_nv'] = np.arange(n_events) + self.event_seen
        self.event_seen = n_events

        return events


def veto_event_dtype(name_event_number='event_number_nv', n_pmts=120):
    dtype = []
    dtype += strax.time_fields  # because mutable
    dtype += [(('Veto event number in this dataset', name_event_number), np.int64),
              (('Last hitlet endtime in event [ns].', 'last_hitlet_endtime'), np.int64),
              (('Total area of all hitlets in evnet [pe]', 'area'), np.float32),
              (('Total number of hitlets in events', 'n_hits'), np.int32),
              (('Area in event per channel [pe]', 'area_per_channel'), np.float32, n_pmts),
              (('Area weighted mean time of the event relative to the event start [ns]',
                'center_time'), np.float32)
              ]
    return dtype


@numba.njit(cache=True, nogil=True)
def compute_event_properties(events, contained_hitlets, start_channel=2000):
    """
    Computes properties of the neutron-veto events. Writes results
    directly to events.

    :param events: Events for which properties should be computed
    :param contained_hitlets: numba.typed.List of hitlets contained in
        each event.
    :param start_channel: Integer specifying start channel, e.g. 2000
        for nveto.
    """
    for e, hitlets in zip(events, contained_hitlets):
        event_area = np.sum(hitlets['area'])
        e['area'] = event_area
        e['n_hits'] = len(hitlets)

        t = hitlets['time'] - hitlets[0]['time']
        if event_area:
            e['center_time'] = np.sum(t * hitlets['area']) / event_area
        else:
            e['center_time'] = np.nan
            
        # Compute endtime of last hitlet in event:
        endtime = strax.endtime(hitlets)
        e['last_hitlet_endtime'] = max(endtime)
        
        # Compute per channel properties:
        for h in hitlets:
            ch = h['channel'] - start_channel
            e['area_per_channel'][ch] += h['area']
