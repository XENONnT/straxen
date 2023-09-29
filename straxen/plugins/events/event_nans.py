import strax
import numpy as np
import straxen

export, __all__ = strax.exporter()

@export
class GiveNans(strax.Plugin):
    """
    A test plugin that just gives nans
    """

    __version__ = '0.0.2'
    
    depends_on = ('events',)
    provides = 'event_nans'
    data_kind = 'events'
    
    dtype = [(('Start time since unix epoch [ns]', 'time'), np.int64),
            (('Exclusive end time since unix epoch [ns]', 'endtime'), np.int64),
            (('Just nans for approximately half of the events', 'event_nans'), np.float64),
            (('Just -1 for approximately half of the events', 'event_negs'), np.int64)]
    
    def compute(self, events):
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']
        
        if len(events)>0:
            halflen = int(len(events)/2)
            result['event_nans'][0] = np.nan
            result['event_negs'][0] = -1
            
        return result