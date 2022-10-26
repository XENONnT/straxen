import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@export
class EventAreaPerChannel(strax.LoopPlugin):
    """
    Simple plugin that provides area per channel, totla (data) and top (data_top) 
    waveforms for main and alternative S1/S2 in the event. 
    """
    depends_on = ('event_basics', 'peaks')
    provides = "event_area_per_channel"
    __version__ = '0.0.2'

    compressor = 'zstd'
    save_when = strax.SaveWhen.EXPLICIT

    n_tpc_pmts = straxen.URLConfig(type=int, help='Number of TPC PMTs')
    n_top_pmts = straxen.URLConfig(default=straxen.n_top_pmts, type=int, help="Number of top PMTs")

    def infer_dtype(self):
        # setting data type from peak dtype
        peak_dtype_dict = {}
        for t_ in strax.peak_dtype(n_channels=self.config['n_tpc_pmts'], digitize_top=True):
            peak_dtype_dict[t_[0][1]]=t_

        ## Populating data type
        infoline = {'s1': 'main S1', 
                    's2': 'main S2', 
                    'alt_s1': 'alternative S1',
                    'alt_s2': 'alternative S2',
                   }
        dtype = []
        # populating APC and waveform samples
        ptypes = ['s1', 's2', 'alt_s1', 'alt_s2']
        for type_ in ptypes:
            dtype +=[((f'Area per channel for {infoline[type_]}', f'{type_}_area_per_channel'),)+
                     peak_dtype_dict['area_per_channel'][1:]]
            dtype +=[((f'Waveform for {infoline[type_]} [ PE / sample ]', f'{type_}_data'),)+
                     peak_dtype_dict['data'][1:]]
            dtype +=[((f'Top waveform for {infoline[type_]} [ PE / sample ]', f'{type_}_data_top'),)+
                     peak_dtype_dict['data_top'][1:]]
            dtype +=[((f'Length of the interval in samples for {infoline[type_]}', f'{type_}_length'),)+
                     peak_dtype_dict['length'][1:]]
            dtype +=[((f'Width of one sample for {infoline[type_]} [ns]', f'{type_}_dt'),)+
                     peak_dtype_dict['dt'][1:]]
        # populating S1 n channel properties
        dtype += [(("Main S1 count of contributing PMTs", "s1_n_channels"),
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
                result[f'{type_}_length'] = peaks['length'][type_index]
                result[f'{type_}_data'] = peaks['data'][type_index]
                result[f'{type_}_data_top'] = peaks['data_top'][type_index]
                result[f'{type_}_dt'] = peaks['dt'][type_index]
                if type_ == 's1':
                    result['s1_n_channels'] = len(type_area_per_channel[type_area_per_channel > 0])
                    result['s1_top_n_channels'] = len(type_area_per_channel[:self.config['n_top_pmts']][
                                                          type_area_per_channel[:self.config['n_top_pmts']] > 0])
                    result['s1_bottom_n_channels'] = len(type_area_per_channel[self.config['n_top_pmts']:][
                                                             type_area_per_channel[self.config['n_top_pmts']:] > 0])
        return result
