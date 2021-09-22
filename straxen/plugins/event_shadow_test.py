import numpy as np
import strax
import straxen

class Shadow(strax.Plugin):
    """Compute several new parameters to describe the noise level.
        We can use these to reduce AC BG. """

    # Name of the data type this plugin provides
    provides = ('event_shadow',) #a tuple, same level as 'event_ifo'etc.

    depends_on = ('event_basics','peak_basics',)

    # Numpy datatype of the output
    dtype = [('shadow', np.float32,'previous s2 shadow at event level [PE/ns]')] + strax.time_fields

    # Version of the plugin. Increment this if you change the algorithm.
    __version__ = '0.0.2'

    def compute(self, events):

        roi_dt = np.dtype([(('back in time', 'time'), int),
                                         (('till it begin','endtime'), int)])
        roi = np.zeros(len(event), dtype=roi_dt)
        
        n_seconds = int(5e9)
        roi['time'] = event['time'] - n_seconds
        roi['endtime'] = event['time']
        split_try = strax.split_by_containment(peak, roi)

        shadow = np.zeros(len(event))
        for event_i, event_a in enumerate(event):
            for peak_i, peak_a in enumerate(split_try[event_i]):
                new_shadow = 0
                if (peak_a['area']>1e3) & (peak_a['type']==2):
                    new_shadow = peak_a['area']/(event_a['time']-peak_a['center_time'])
                    
                if new_shadow > shadow[event_i]:
                    shadow[event_i] = new_shadow
       
        return dict(time=events['time'],
            endtime=strax.endtime(events),
                    shadow = shadow)