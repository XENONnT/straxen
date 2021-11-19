import numpy as np
import strax

export, __all__ = strax.exporter()

@export
class EventShadow(strax.Plugin):
    """
    This plugin can calculate shadow at event level.
    It depends on peak-level shadow.
    The event-level shadow is its first S2 peak's shadow.
    If no S2 peaks, the shadow will be nan. 
    It also gives the position infomation of the previous S2s.
    """
    __version__ = '0.0.7'
    depends_on = ('event_basics', 'peak_basics', 'peak_shadow')
    provides = 'event_shadow'
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = [('shadow', np.float32, 'previous s2 shadow [PE/ns]'),
                 ('pre_s2_area', np.float32, 'previous s2 area [PE]'),
                 ('shadow_dt', np.int64, 'time diffrence to the previous s2 [ns]'),
                 ('shadow_index', np.int32, 'max shadow peak index in event'),
                 ('pre_s2_x', np.float32, 'x of previous s2 peak causing shadow [cm]'),
                 ('pre_s2_y', np.float32, 'y of previous s2 peak causing shadow [cm]'),
                 ('shadow_distance', np.float32, 'distance to the s2 peak with max shadow [cm]')]
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        res = np.zeros(len(events), self.dtype)
        # Initialization
        res['shadow_index'] = -1
        res['pre_s2_x'] = np.nan
        res['pre_s2_y'] = np.nan
        for event_i, sp in enumerate(split_peaks):
            if (sp['type'] == 2).sum() > 0:
                # Define event shadow as the first S2 peak shadow
                first_s2_index = np.argwhere(sp['type'] == 2)[0]
                res['shadow_index'][event_i] = first_s2_index
                res['shadow'][event_i] = sp['shadow'][first_s2_index]
                res['pre_s2_area'][event_i] = sp['pre_s2_area'][first_s2_index]
                res['shadow_dt'][event_i] = sp['shadow_dt'][first_s2_index]
                res['pre_s2_x'][event_i] = sp['pre_s2_x'][first_s2_index]
                res['pre_s2_y'][event_i] = sp['pre_s2_y'][first_s2_index]
        res['shadow_distance'] = ((res['pre_s2_x'] - events['s2_x'])**2+(res['pre_s2_y'] - events['s2_y'])**2)**0.5
        res['time'] = events['time']
        res['endtime'] = strax.endtime(events)
        return res
