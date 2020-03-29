import strax
import straxen
import numba
import numpy as np
from numba import njit
from frozendict import frozendict

export, __all__ = strax.exporter()

@strax.takes_config(
    # V1495 busy module:
    # Generates a 25 ns NIM pulse whenever a veto begins and a 25 ns NIM signal when it ends. 
    # A new start signal can occur only after the previous busy instance ended.
    # 1ms (1e6 ns) - minimum busy veto length, or until the board clears its memory
    
    # DDC10 High Energy Veto:
    # 10ms (1e7 ns) - fixed HE veto length in XENON1T DDC10, 
    # in XENONnT it will be calibrated based length of large S2 SE tails
    
    strax.Option('max_veto_gap', default = int(5e8),
                 help='Maximum separation between veto stop and start pulses [ns]'),
    strax.Option('channel_map', track=False, type=frozendict, 
                 help="frozendict mapping subdetector to (min, max)"
                      "channel number."))

@export
class VetoIntervals(strax.OverlapWindowPlugin):
    """ Merge start and stop times in acquisition monitor data 
    for the low energy and high energy channels, as well as for the high energy veto 
    no tag  <= V1495 busy veto for tpc channels
    bb tag  <= V1495 busy veto for high energy tpc channels
    hev tag <= DDC10 hardware high energy veto
    """
        
    __version__ ='0.0.2'
    depends_on = ('raw_records_aqmon') 
    provides  = ('veto_intervals',)
    data_kind = ('veto_intervals',)

    aqmon_channel_names = ('sum_wf','m_veto_sync','hev_stop', 'hev_start', 'bb_stop', 'bb_start','stop', 'start')
    duration_dtype = [(('Interval between start and stop in [ns]', 'interval'), np.int64)]
    
    # Using the time & endtime dtypes for busy so strax does not complain about lack of endtime and time dtypes
    def infer_dtype(self):
        dtype = []
        for veto in ('','hev_','bb_'):
            dtype += [(('Start '+ veto +'veto time since unix epoch [ns]', veto+'time'), np.int64)]
            dtype += [(('Stop '+ veto +'veto time since unix epoch [ns]', veto+'endtime'), np.int64)]
            dtype += [(('Duration of '+ veto +'veto time since unix epoch [ns]', veto+'interval'), np.int64)]
        return dtype
    
    def setup(self):
        self.channel_range = self.config['channel_map']['aqmon']
        self.channel_numbers = np.arange(self.channel_range[0]+1,self.channel_range[1]+1,1)
        self.channel_map = dict(zip(self.aqmon_channel_names,self.channel_numbers))
        return self.channel_map
    
    def get_window_size(self):
        return (self.config['max_veto_gap'])    
    
    
    def compute(self, raw_records_aqmon): 
        r = raw_records_aqmon
        vetos = dict()
        
        for i, v in enumerate(['','bb_','hev_']):
            channels = channel_select(r,self.channel_map[v+'stop'], self.channel_map[v +'start'])
            vetos[v +'veto'] = merge_vetos(channels,gap = self.config['max_veto_gap'],\
                                           dtype = strax.time_fields + self.duration_dtype)
            result = strax.dict_to_rec(vetos)
        
        busy_interval = result['veto']
        bb_interval = result['bb_veto']
        hev_interval = result['hev_veto']
        
        return dict(time = busy_interval['time'],
                   endtime = busy_interval['endtime'],
                   interval = busy_interval['interval'],
                   hev_time = hev_interval['time'],
                   hev_endtime = hev_interval['endtime'],
                   hev_interval = hev_interval['interval'],
                   bb_time = bb_interval['time'],
                   bb_endtime = bb_interval['endtime'],
                   bb_interval = bb_interval['interval'])
    
    @staticmethod   
    def merge_vetos(channels, gap, dtype):
        if len(channels):
            start, stop = strax.find_peak_groups(channels, gap_threshold = gap)  
            result = np.zeros(len(start),dtype=dtype)
            result['time'] = start
            result['endtime'] = stop
            result['interval'] = stop - start
        else:
            result = np.zeros(1,dtype=dtype)
            result['time'] = 0
            result['endtime'] = 0 
            result['interval'] = 0
            return result
            
@numba.njit
def _mask(x, mask):
    return x[mask]

@export
@numba.njit
def channel_select(rr, ch_stop, ch_start):
    """Return """
    return _mask(rr, (rr['channel'] >= ch_stop) & (rr['channel'] <= ch_start))      
