import numba
from straxen.plugins.position_reconstruction import DEFAULT_POSREC_ALGO
from straxen.common import pax_file, get_resource, first_sr1_run, rotate_perp_wires
from straxen.get_corrections import get_cmt_resource, is_cmt_option
from straxen.itp_map import InterpolatingMap
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
    provides = 'peaksevent'
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
class PeakCorrectedAreas(strax.Plugin):
    """
    Plugin which applies light collection efficiency maps and electron
    life time to the data.
    Computes the cS2 for all s2 peaks inside drift length. 
    Note:
    Assumes main s1 is the only valid one. 
    Future work needed:
    Energy as a function of cs2.
    """
    __version__ = '0.0.1'

    depends_on = ['peak_basics','peak_positions','peaks_per_event']
    data_kind = 'peaks'
    provides = 'peaks_corrections'
    # Descriptor configs
    elife = straxen.URLConfig(
        default='cmt://elife?version=ONLINE&run_id=plugin.run_id',
        help='electron lifetime in [ns]')

    # default posrec, used to determine which LCE map to use
    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO,
        help="default reconstruction algorithm that provides (x,y)"
    )
    s1_xyz_map = straxen.URLConfig(
        default='itp_map://resource://cmt://format://'
                's1_xyz_map_{algo}?version=ONLINE&run_id=plugin.run_id'
                '&fmt=json&algo=plugin.default_reconstruction_algorithm',
        cache=True)
    s2_xy_map = straxen.URLConfig(
        default='itp_map://resource://cmt://format://'
                's2_xy_map_{algo}?version=ONLINE&run_id=plugin.run_id'
                '&fmt=json&algo=plugin.default_reconstruction_algorithm',
        cache=True)

    # average SE gain for a given time period. default to the value of this run in ONLINE model
    # thus, by default, there will be no time-dependent correction according to se gain
    avg_se_gain = straxen.URLConfig(
        default='cmt://avg_se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Nominal single electron (SE) gain in PE / electron extracted. '
             'Data will be corrected to this value')

    # se gain for this run, allowing for using CMT. default to online
    se_gain = straxen.URLConfig(
        default='cmt://se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Actual SE gain for a given run (allows for time dependence)')

    # relative extraction efficiency which can change with time and modeled by CMT.
    rel_extraction_eff = straxen.URLConfig(
        default='cmt://rel_extraction_eff?version=ONLINE&run_id=plugin.run_id',
        help='Relative extraction efficiency for this run (allows for time dependence)')

    # relative light yield
    # defaults to no correction
    rel_light_yield = straxen.URLConfig(
        default='cmt://relative_light_yield?version=ONLINE&run_id=plugin.run_id',
        help='Relative light yield (allows for time dependence)'
    )
    
    region_linear = straxen.URLConfig(
        default=28,
        help='linear cut (cm) for ab region, check out the note https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:sr0_2_region_se_correction'
    )
    
    region_circular = straxen.URLConfig(
        default=60,
        help='circular cut (cm) for ab region, check out the note https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:sr0_2_region_se_correction'
    )

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
    
    def ab_region(self, x, y):
        new_x, new_y = rotate_perp_wires(x, y)
        cond = new_x < self.region_linear
        cond &= new_x > -self.region_linear
        cond &= new_x**2 + new_y**2 < self.region_circular**2
        return cond
    
    def cd_region(self, x, y):
        return ~self.ab_region(x, y)

    def compute(self, peaks):
        result = np.zeros(len(peaks), self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = peaks['endtime']

        # S1 corrections depend on the actual corrected event position.
        # We use this also for the alternate S1; for e.g. Kr this is
        # fine as the S1 correction varies slowly.
 

        # s2 corrections
        # S2 top and bottom are corrected separately, and cS2 total is the sum of the two
        # figure out the map name
        if len(self.s2_xy_map.map_names) > 1:
            s2_top_map_name = "map_top"
            s2_bottom_map_name = "map_bottom"
        else:
            s2_top_map_name = "map"
            s2_bottom_map_name = "map"

        regions = {'ab': self.ab_region, 'cd': self.cd_region}

        # setup SEG and EE corrections
        # if they are dicts, we just leave them as is
        # if they are not, we assume they are floats and
        # create a dict with the same correction in each region
        if isinstance(self.se_gain, dict):
            seg = self.se_gain
        else:
            seg = {key: self.se_gain for key in regions}

        if isinstance(self.avg_se_gain, dict):
            avg_seg = self.avg_se_gain
        else:
            avg_seg = {key: self.avg_se_gain for key in regions}

        if isinstance(self.rel_extraction_eff, dict):
            ee = self.rel_extraction_eff
        else:
            ee = {key: self.rel_extraction_eff for key in regions}

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

        for partition, func in regions.items():
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
   

@export
class EventInfoMS(strax.Plugin):
    """
    Get the sum of s2 and s1,
    Get the sum of cs2 inside the drift length. 
    """
    version = '0.0.1'
    depends_on = ('events', 'peak_basics','peaks_per_event','peaks_corrections')
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
            cond = (sp["type"]==2)&(sp["drift_time"]>0)&(sp["drift_time"]< 1.01 * self.drift_time_max)
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
