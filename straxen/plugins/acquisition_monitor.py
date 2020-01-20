import os
import numba
import strax
import numpy as np
from numba import njit

export, __all__ = strax.exporter()

# TODO: Connect to mongoDB and output times via bootstrax & its context
@export
@strax.takes_config(
    strax.Option('run_start_time', default=.0, type=float, track=False,
                 help="time of start run (s since unix epoch)"))


class VetoTimesRecorder(strax.Plugin):
    """
    Puts the aqmon times for busy on/off into a dict to be further put
    into MongoDB and used by the deadtime monitor in NODIAQ
    """
    __version__ = '0.0.1'
    
    rechunk_on_save = False
    parallel ='process'

    veto_signals = [(('Start time of high E ch. busy (ns since unix epoch)','busy_start'), np.int64),
                    (('Stop time of high E ch. busy (ns since unix epoch)','busy_stop'), np.int64),
                    (('Start time of HEV (ns since unix epoch)','hev_start'), np.int64),
                    (('Stop time of HEV (ns since unix epoch)','hev_stop'), np.int64)]

    depends_on = ('aqmon_records')
    provides = ('busy_start', 'busy_stop', 'hev_start', 'hev_stop')
    data_kind = {k: k for k in provides}

    # TODO: Include also the the n_veto busy_start/stop signals
    # TODO: XENONnT acq_mon channels will be different, get them from config
    veto_channels = list(range(255,258+1))
    ch_map = dict(zip(provides,veto_channels))
    
    
    def infer_dtype(self):
        dtype = dict()
        for i, p in enumerate(self.provides):
            dtype[p] = self.veto_signals[i]   
        return dtype
    
    def compute(self, aqmon_records):
        veto_times = dict()
        
        raw = aqmon_records
        r = raw[(raw['channel'] >= min(self.veto_channels)) & (raw['channel'] <= max(self.veto_channels)+1)]
        for ind, v in enumerate(self.provides):
            veto_times[v] = channel_select(r, self.ch_map[v])['time'] - self.config['run_start_time']    
       
        return dict(veto_times = veto_times)
    
@numba.njit
def _mask(x, mask):
    return x[mask]


@export
@numba.njit
def channel_select(rr, ch):
    """Return """
    return _mask(rr, rr['channel'] == ch)
