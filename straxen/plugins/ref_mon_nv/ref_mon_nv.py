import numpy as np
from immutabledict import immutabledict

import strax
import straxen

from itertools import chain

export, __all__ = strax.exporter()


@export
class nVeto_reflectivity(strax.Plugin):
    '''
    Plugin which computes parameters used for the Reflectivity
    Monitor for the Neutron Veto
    
    From the raw records, the start time of each trigger
    window is extracted (by taking the smallest "time" of all the channels)
    
    Using the hitlets, the delta times wrt the beginning of the trigger
    are computed
    
    Cuts can be made in area (of all the hitlets), time (since the beginning
    of the trigger window) and number of contributing PMTs (in each window)
    
    The returned values are the delta times (used for the reflectivity), PMT
    channels and areas
    
    '''
    __version__ = '0.1.0'

    parallel = 'process'
    rechunk_on_save = True
    compressor = 'zstd'

    depends_on = ('raw_records_nv','hitlets_nv')

    provides = 'ref_mon_nv'
    data_kind = 'ref_mon_nv'


    dtype = strax.time_fields + [(('Time with respect to the precedent trigger [ns]','time_since_trigger'),np.int32),
             (('Channel of the hit','channel'),np.int16),
             (('Area of the hit [PE]','area'),np.float32),
             (('Number of event','number'),np.int32)]
    
    
    def compute(self,raw_records_nv,hitlets_nv):

        split_time = 5000     
            
        # find the trigger times
        time_unique = np.unique(raw_records_nv['time'])
        time_diff = np.diff(time_unique,prepend=time_unique[0])
        time_unique_split = np.split(time_unique,np.where(time_diff>=split_time)[0])
        trigger_times = np.array([i[0] for i in time_unique_split])

        # split the hits in trigger windows
        difference = np.diff(hitlets_nv['time'],prepend=hitlets_nv['time'][0])
        hits_split = np.split(hitlets_nv,np.where(difference>split_time)[0])
        numbers = [ np.repeat(i,len(hits_split[i])) for i in range(len(hits_split)) ]
                
        # compute the time differences        
        times_wrt_trigger = [ hits_split[i]['time'] - trigger_times[np.argmin(np.abs(trigger_times-hits_split[i][0]['time']))] for i in range(len(hits_split)) ]
        

        # reshape the arrays
        hits_split = np.array(list(chain.from_iterable(hits_split)))
        times_wrt_trigger = np.array(list(chain.from_iterable(times_wrt_trigger)))
        numbers = np.array(list(chain.from_iterable(numbers)))
   
        # fill the delta_times 
        delta_times = np.zeros(len(hits_split),dtype=self.dtype)
        delta_times['time'] = hits_split['time']
        delta_times['endtime'] = hits_split['time'] + hits_split['length'] * hits_split['dt']
        delta_times['time_since_trigger'] = times_wrt_trigger
        delta_times['area'] = hits_split['area']
        delta_times['channel'] = hits_split['channel']
        delta_times['number'] = numbers
        
        
        return delta_times