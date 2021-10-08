import numpy as np
import strax
import numba
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('pre_s2_area_threshold', default=1000,
                 help='Only take S2s large than this into account when calculating EventShadow.'),
    strax.Option('time_window_backward', default=int(3e9),
                 help='Search for S2s causing shadow in this time window'))
class EventShadow(strax.Plugin):
    """
    This plugin can find and calculate the previous S2 shadow at event level,
    with time window backward and previous S2 area as options.
    It also gives the area and position infomation of these previous S2s.
    """
    __version__ = '0.0.6'
    depends_on = ('event_basics','peak_basics','peak_positions')
    provides = 'event_shadow'
    save_when = strax.SaveWhen.EXPLICIT
    

    def infer_dtype(self):
        dtype = [
                ('pre_s2_area', np.float32,'previous s2 area [PE]'),
                 ('shadow_dt', np.int64,'time diffrence to the previous s2 [ns]'),
                 ('shadow', np.float32,'previous s2 shadow [PE/ns]'),
                ('pre_s2_x', np.float32,'x of previous s2 peak causing shadow [cm]'),
                 ('pre_s2_y', np.float32,'y of previous s2 peak causing shadow [cm]'),
                 ('shadow_distance', np.float32,'distance to the previous s2 peak causing the max shadow [cm]')           
                ]
        dtype += strax.time_fields
        return dtype
    
    def compute(self, events, peaks):
        roi_dt = np.dtype([(('back in time', 'time'), int),
                                         (('till it begin','endtime'), int)])
        roi = np.zeros(len(events), dtype=roi_dt)   
        n_seconds = self.config['time_window_backward']
        roi['time'] = events['time'] - n_seconds
        roi['endtime'] = events['time']
        mask_s2 = peaks['type'] == 2
        mask_s2  &= peaks['area'] > self.config['pre_s2_area_threshold']   
        split_peaks = strax.split_touching_windows(peaks[mask_s2], roi)
        res = np.zeros(len(events), self.dtype)       
        compute_shadow(events, split_peaks, res)
        res['shadow_distance'] = ((res['pre_s2_x'] - events['s2_x'])**2+(res['pre_s2_y'] - events['s2_y'])**2)**0.5
        res['time'] = events['time']
        res['endtime'] = strax.endtime(events)
        
        return res
    
def compute_shadow(events, split_peaks, res):
    if len(res):
        return _compute_shadow(events, split_peaks, res)


@numba.njit(cache=True)
def _compute_shadow(events, split_peaks, res):
    for event_i, event_a in enumerate(events):
        new_shadow = 0
        for peak_i, peak_a in enumerate(split_peaks[event_i]):
            new_shadow = peak_a['area']/(event_a['s2_center_time']-peak_a['center_time'])
            if new_shadow > res['shadow'][event_i]:
                res['pre_s2_area'][event_i] = peak_a['area']
                res['shadow_dt'][event_i] = event_a['s2_center_time']-peak_a['center_time']
                res['pre_s2_x'][event_i] = peak_a['x']
                res['pre_s2_y'][event_i] = peak_a['y']
                res['shadow'][event_i] = new_shadow     
