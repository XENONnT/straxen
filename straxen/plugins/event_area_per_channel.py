import numpy as np
import strax
import straxen
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('n_tpc_pmts', type=int,
                 help='Number of TPC PMTs'),
)
class EventAreaPerChannel(strax.LoopPlugin):
    """
    Simple plugin that provides area per channel for main 
    and alternative S1/S2 in the event. 
    """
    depends_on = ('event_basics', 'peaks')
    provides = "event_area_per_channel"
    __version__ = '0.0.0'
    def infer_dtype(self):
        dtype = []
        dtype.append( (("Area per channel for main S2", "s2_area_per_channel"), 
                       np.float32, (self.config['n_tpc_pmts'],)) )
        dtype.append( (("Area per channel for alternative S2", "alt_s2_area_per_channel"), 
                       np.float32, (self.config['n_tpc_pmts'],)) )
        dtype.append( (("Area per channel for main S1", "s1_area_per_channel"), 
                       np.float32, (self.config['n_tpc_pmts'],)) )
        dtype.append( (("Area per channel for alternative S1", "alt_s1_area_per_channel"), 
                       np.float32, (self.config['n_tpc_pmts'],)) )
        dtype += strax.time_fields
        return dtype
    
    def compute_loop(self, event, peaks):
        result = dict()
        result['time'], result['endtime'] = event['time'], strax.endtime(event)
        for type_ in ['s1','s2','alt_s1','alt_s2']:
            result[type_+'_area_per_channel'] = (peaks['area_per_channel'][event[type_+"_index"]] if event[type_+"_index"]!=-1 else 0.0)
        return result
