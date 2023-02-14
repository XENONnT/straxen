import strax
import numpy as np

export, __all__ = strax.exporter()


@export
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
        return 120 * 10 ** 9

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
