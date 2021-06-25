import numpy as np
import strax
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

    compressor = 'zstd'
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = [(("Area per channel for main S2", "s2_area_per_channel"), 
                  np.float32, (self.config['n_tpc_pmts'],)),
                 (("Area per channel for alternative S2", "alt_s2_area_per_channel"),
                  np.float32, (self.config['n_tpc_pmts'],)),
                 (("Area per channel for main S1", "s1_area_per_channel"), 
                  np.float32, (self.config['n_tpc_pmts'],)),
                 (("Area per channel for alternative S1", "alt_s1_area_per_channel"), 
                  np.float32, (self.config['n_tpc_pmts'],)),
                ]
        dtype += strax.time_fields
        return dtype
    
    def compute_loop(self, event, peaks):
        result = dict()
        result['time'] = event['time']
        result['endtime'] = strax.endtime(event)
        
        for type_ in ['s1', 's2', 'alt_s1', 'alt_s2']:
            type_index = event[f'{type_}_index']
            if type_index != -1:
                type_ara_per_channel = peaks['area_per_channel'][type_index]
                result[f'{type_}_area_per_channel'] = type_ara_per_channel
        return result
