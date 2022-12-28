import numba

from straxen.common import pax_file, get_resource, first_sr1_run, rotate_perp_wires
from straxen.get_corrections import get_cmt_resource, is_cmt_option
from straxen.itp_map import InterpolatingMap
import strax
import straxen
import numpy as np
export, __all__ = strax.exporter()

@export
class EventInfoMS(strax.Plugin):
    """
    Get the sum of s2 and s1,
    Get the sum of cs2 inside the drift length. 
    """
    __version__ = '0.0.1'
    depends_on = ('event_info', 'peak_basics','peaks_per_event','peaks_corrections')
    provides = 'event_MS_naive'


    
    def infer_dtype(self):
        dtype = []
        dtype += [
            ((f's1_sum'), np.float32),
            ((f's2_sum'), np.float32),
            ((f'cs2_sum'), np.float32),
            ((f'cs2_wo_timecorr_sum'), np.float32),
            ((f'cs2_wo_elifecorr_sum'), np.float32),
            ((f'cs2_area_fraction_sum'), np.float32),
            ((f'ces_sum'), np.float32),
            ((f'e_charge_sum'), np.float32),
             ]
        dtype += strax.time_fields
        return dtype

    save_when = strax.SaveWhen.TARGET

    # config options don't double cache things from the resource cache!
    g1 = straxen.URLConfig(
        default='bodega://g1?bodega_version=v2',
        help="S1 gain in PE / photons produced",
    )
    g2 = straxen.URLConfig(
        default='bodega://g2?bodega_version=v2',
        help="S2 gain in PE / electrons produced",
    )
    lxe_w = straxen.URLConfig(
        default=13.7e-3,
        help="LXe work function in quanta/keV"
    )
    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )
    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z, infer_type=False,
        help='Total length of the TPC from the bottom of gate to the '
             'top of cathode wires [cm]', )
    def setup(self):
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)
    def cs1_to_e(self, x):
        return self.lxe_w * x / self.g1

    def cs2_to_e(self, x):
        return self.lxe_w * x / self.g2

    
    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.infer_dtype())
        #result['s2_sum'] = np.zeros(len(events))
   
         #1. Assign peaks features to main S1 and main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            cond = (sp["type"]==2)&(sp["drift_time"]>0)&(sp["drift_time"]< 1.01 * self.drift_time_max)&(sp["cs2"]>0)
            result[f's2_sum'][event_i] = np.sum(sp[cond]['area'])
            result[f'cs2_sum'][event_i] = np.sum(sp[cond]['cs2'])
            result[f'cs2_wo_timecorr_sum'][event_i] = np.sum(sp[cond]['cs2_wo_timecorr'])
            result[f'cs2_wo_elifecorr_sum'][event_i] = np.sum(sp[cond]['cs2_wo_elifecorr'])
            result[f'cs2_area_fraction_sum'][event_i] = np.sum(sp[cond]['cs2_area_fraction_top'])            
            result[f's1_sum'][event_i] = np.sum(sp[sp["type"]==1]['area'])
        el = self.cs1_to_e(events[f'cs1'])
        ec = self.cs2_to_e(result[f'cs2_sum'])
        result[f'ces_sum'] = el+ec
        result[f'e_charge_sum'] = ec
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        return result
