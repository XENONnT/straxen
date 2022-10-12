import strax
import straxen

from straxen.plugins.events_nv.events_nv import veto_event_dtype, nVETOEvents

export, __all__ = strax.exporter()


@export
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


