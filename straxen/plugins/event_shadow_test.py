import numpy as np
import strax
import straxen

class Shadow(strax.Plugin):
    """Compute several new parameters to describe the noise level.
        We can use these to reduce AC BG. """

    # Name of the data type this plugin provides
    provides = ('event_shadow',) #a tuple, same level as 'event_ifo'etc.

    depends_on = ('event_basics',)

    # Numpy datatype of the output
    dtype = [('shadow', np.float32,'previous s2 shadow at event level [PE/ns]')] + strax.time_fields

    # Version of the plugin. Increment this if you change the algorithm.
    __version__ = '0.0.1'

    def compute(self, events):

        shadow = np.zeros(len(events))
        pre_s2_x = np.zeros(len(events))
        pre_s2_y = np.zeros(len(events))
        distance_to_pre_s2 = np.zeros(len(events))
        len_events = len(events)

        for i in range(1,20):
            new_shadow = events['s2_area'][0:len_events-i]/(events['s2_center_time'][i:]-events['s2_center_time'][0:len_events-i])
            test = new_shadow > shadow[i:]
            pre_s2_x[i:]=events['s2_x'][0:len_events-i]*test + pre_s2_x[i:]*(~test)
            pre_s2_y[i:]=events['s2_y'][0:len_events-i]*test + pre_s2_y[i:]*(~test)
            shadow[i:] = new_shadow*test + shadow[i:]*(~test)

        #distance_to_pre_s2 = pre_s2_x**2 + pre_s2_y**2
        
        return dict(time=events['time'],
            endtime=strax.endtime(events),
                    shadow = shadow)