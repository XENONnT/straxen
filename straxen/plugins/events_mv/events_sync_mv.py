import numpy as np
import strax
import straxen

from straxen.plugins.events_nv import nVETOEventsSync

import strax

export, __all__ = strax.exporter()


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
