import numpy as np
import pandas as pd
import strax
import straxen
import warnings
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
    depends_on = ('event_basics','peak_basics','peak_positions')
    parallel = False
    provides = 'event_shadow'
    __version__ = '0.0.6'

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
        #set a time window to look for peaks
        roi_dt = np.dtype([(('back in time', 'time'), int),
                                         (('till it begin','endtime'), int)])
        roi = np.zeros(len(events), dtype=roi_dt)   
        n_seconds = self.config['time_window_backward']
        roi['time'] = events['time'] - n_seconds
        roi['endtime'] = events['time']
        split_try = strax.split_touching_windows(peaks, roi)
        #define the variables we want to calculate
        pre_s2_area = np.zeros(len(events))
        shadow_dt = np.zeros(len(events))     
        pre_s2_x = np.zeros(len(events))
        pre_s2_y = np.zeros(len(events))
        shadow = np.zeros(len(events)) 
        shadow_distance = np.zeros(len(events))   
        
        #loop over interested peaks for each event
        
        #this part should be speed up
        for event_i, event_a in enumerate(events):
            for peak_i, peak_a in enumerate(split_try[event_i]):
                new_shadow = 0
                if  (peak_a['area']>self.config['pre_s2_area_threshold'])&(peak_a['type']==2):
                    new_shadow = peak_a['area']/(event_a['time']-peak_a['center_time'])

                if new_shadow > shadow[event_i]:
                    pre_s2_area[event_i] = peak_a['area']
                    shadow_dt[event_i] = event_a['time']-peak_a['center_time']
                    pre_s2_x[event_i] = peak_a['x']
                    pre_s2_y[event_i] = peak_a['y']
                    shadow[event_i] = new_shadow
                    
        shadow_distance = ((pre_s2_x - events['s2_x'])**2+(pre_s2_y - events['s2_y'])**2)**0.5

        return dict(time=events['time'],
            endtime=strax.endtime(events),
                    shadow = shadow,
                    pre_s2_area = pre_s2_area,
                    shadow_dt = shadow_dt,
                    pre_s2_x = pre_s2_x,
                    pre_s2_y = pre_s2_y,
                    shadow_distance = shadow_distance)
    
