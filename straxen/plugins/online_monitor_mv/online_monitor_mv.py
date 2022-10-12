import strax
import numpy as np
from straxen.plugins.online_monitor_nv.online_monitor_nv import OnlineMonitorNV, veto_monitor_dtype

export, __all__ = strax.exporter()


@export
class OnlineMonitorMV(OnlineMonitorNV):
    __doc__ = OnlineMonitorNV.__doc__.replace('_nv', '_mv').replace('nVeto', 'muVeto')
    depends_on = ('hitlets_mv', 'events_mv')
    provides = 'online_monitor_mv'
    data_kind = 'online_monitor_mv'
    rechunk_on_save = False

    # Needed in case we make again an muVETO child.
    ends_with = '_mv'
    child_plugin = True

    __version__ = '0.0.2'

    def infer_dtype(self):
        self.channel_range = self.channel_map['mv']
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1
        return veto_monitor_dtype(self.ends_with, self.n_channel, self.events_area_nbins)

    def compute(self, hitlets_mv, events_mv, start, end):
        events_mv = np.copy(events_mv)
        return super().compute(hitlets_mv, events_mv, start, end)
