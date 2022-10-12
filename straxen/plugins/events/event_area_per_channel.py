import numpy as np
import strax
import straxen
export, __all__ = strax.exporter()

@export
class EventAreaPerChannel(strax.LoopPlugin):
    """
    Simple plugin that provides area per channel for main 
    and alternative S1/S2 in the event. 
    """
    depends_on = ('event_basics', 'peaks')
    provides = "event_area_per_channel"
    __version__ = '0.0.1'

    compressor = 'zstd'
    save_when = strax.SaveWhen.EXPLICIT

    n_tpc_pmts = straxen.URLConfig(type=int, help='Number of TPC PMTs')
    n_top_pmts = straxen.URLConfig(default=straxen.n_top_pmts, type = int, help="Number of top PMTs")

    def infer_dtype(self):
        dtype = [(("Area per channel for main S2", "s2_area_per_channel"), 
                  np.float32, (self.config['n_tpc_pmts'],)),
                 (("Area per channel for alternative S2", "alt_s2_area_per_channel"),
                  np.float32, (self.config['n_tpc_pmts'],)),
                 (("Area per channel for main S1", "s1_area_per_channel"), 
                  np.float32, (self.config['n_tpc_pmts'],)),
                 (("Area per channel for alternative S1", "alt_s1_area_per_channel"), 
                  np.float32, (self.config['n_tpc_pmts'],)),
                 (("Main S1 count of contributing PMTs", "s1_n_channels"), 
                  np.int16),
                 (("Main S1 top count of contributing PMTs", "s1_top_n_channels"), 
                  np.int16),
                 (("Main S1 bottom count of contributing PMTs", "s1_bottom_n_channels"), 
                  np.int16),
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
                type_area_per_channel = peaks['area_per_channel'][type_index]
                result[f'{type_}_area_per_channel'] = type_area_per_channel
                
                if type_ == 's1':
                    result['s1_n_channels'] = len(type_area_per_channel[type_area_per_channel > 0])
                    result['s1_top_n_channels'] = len(type_area_per_channel[:self.config['n_top_pmts']][type_area_per_channel[:self.config['n_top_pmts']] > 0])
                    result['s1_bottom_n_channels'] = len(type_area_per_channel[self.config['n_top_pmts']:][type_area_per_channel[self.config['n_top_pmts']:] > 0])
        return result
