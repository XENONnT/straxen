import numba
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO
from straxen.common import pax_file, get_resource, first_sr1_run, rotate_perp_wires
from straxen.get_corrections import get_cmt_resource, is_cmt_option
from straxen.itp_map import InterpolatingMap
from straxen.plugins.events.corrected_areas import CorrectedAreas
import strax
import straxen
import numpy as np
export, __all__ = strax.exporter()

@export
class EventPeaks(strax.Plugin):
    """
    Add event number for peaks and drift times of all s2 depending on the largest s1.
    Link - https://xe1t-wiki.lngs.infn.it/doku.php?id=weiss:analysis:ms_plugin
    """
    __version__ = '0.0.1'
    depends_on = ('event_basics', 'peak_basics','peak_positions')
    provides = 'peaks_per_event'
    data_kind = 'peaks'

    def infer_dtype(self):
        dtype = []
        dtype += [
            ((f'event_number'), np.float32),
            ((f'drift_time'), np.float32),
             ]
        dtype += strax.time_fields
        return dtype

    save_when = strax.SaveWhen.TARGET
 
    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        split_peaks_ind = strax.fully_contained_in(peaks, events)
        result = np.zeros(len(peaks), self.infer_dtype())
        result.fill(np.nan)
        #result['s2_sum'] = np.zeros(len(events))
   
         #1. Assign peaks features to main S1 and main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            result[f'drift_time'][(split_peaks_ind==event_i)] = sp[f'center_time']-event[f's1_center_time']
        result[f'event_number']= split_peaks_ind
        result[f'drift_time'][peaks["type"]!=2]= np.nan
        result['time'] = peaks['time']
        result['endtime'] = strax.endtime(peaks)
        return result


@export
class PeakCorrectedAreas(CorrectedAreas):
    
    __version__ = '0.0.0'

    depends_on = ['peak_basics','peak_positions','peaks_per_event']
    data_kind = 'peaks'
    provides = 'peaks_corrections'
    
    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        dtype += [
                      (f'cs2_wo_elifecorr', np.float32,
                       f'Corrected area of S2 before elife correction '
                       f'(s2 xy correction + SEG/EE correction applied) [PE]'),
                      (f'cs2_wo_timecorr', np.float32,
                       f'Corrected area of S2 before SEG/EE and elife corrections'
                       f'(s2 xy correction applied) [PE]'),
                      (f'cs2_area_fraction_top', np.float32,
                       f'Fraction of area seen by the top PMT array for corrected  S2'),
                      (f'cs2_bottom', np.float32,
                       f'Corrected area of S2 in the bottom PMT array [PE]'),
                      (f'cs2', np.float32, f'Corrected area of  S2 [PE]'), ]
        return dtype
    
    def compute(self, peaks):
        result = np.zeros(len(peaks), self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = peaks['endtime']

        # s2 corrections
        s2_top_map_name, s2_bottom_map_name = self.s2_map_names()
        
        seg, avg_seg, ee = self.SEG_EE_correction_preparation()
        # now can start doing corrections

        # S2(x,y) corrections use the observed S2 positions
        s2_positions = np.vstack([peaks[f'x'], peaks[f'y']]).T

        # corrected s2 with s2 xy map only, i.e. no elife correction
        # this is for s2-only events which don't have drift time info

        cs2_top_xycorr = (peaks[f'area']
                          * peaks[f'area_fraction_top']
                          / self.s2_xy_map(s2_positions, map_name=s2_top_map_name))
        cs2_bottom_xycorr = (peaks[f'area']
                             * (1 - peaks[f'area_fraction_top'])
                             / self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name))

        # For electron lifetime corrections to the S2s,
        # use drift time computed using the main S1.
        
        elife_correction = np.exp(peaks[f'drift_time'] / self.elife)
        result[f"cs2_wo_timecorr"] = ((cs2_top_xycorr + cs2_bottom_xycorr) * elife_correction)

        for partition, func in self.regions.items():
            # partitioned SE and EE
            partition_mask = func(peaks[f'x'], peaks[f'y'])

            # Correct for SEgain and extraction efficiency
            seg_ee_corr = seg[partition]/avg_seg[partition]*ee[partition]

            # note that these are already masked!
            cs2_top_wo_elifecorr = cs2_top_xycorr[partition_mask] / seg_ee_corr
            cs2_bottom_wo_elifecorr = cs2_bottom_xycorr[partition_mask] / seg_ee_corr

            result[f"cs2_wo_elifecorr"][partition_mask] = cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr

            # cs2aft doesn't need elife/time corrections as they cancel
            result[f"cs2_area_fraction_top"][partition_mask] = cs2_top_wo_elifecorr / (cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr)

            result[f"cs2"][partition_mask] = result[f"cs2_wo_elifecorr"][partition_mask] * elife_correction[partition_mask]
            result[f"cs2_bottom"][partition_mask] = cs2_bottom_wo_elifecorr * elife_correction[partition_mask]
        result[f"cs2_wo_timecorr"][peaks["type"]!=2] = np.nan
        result[f"cs2_wo_elifecorr"][peaks["type"]!=2] = np.nan
        result[f"cs2_area_fraction_top"][peaks["type"]!=2] = np.nan
        result[f"cs2"][peaks["type"]!=2] = np.nan
        result[f"cs2_bottom"][peaks["type"]!=2] = np.nan
        return result