import numpy as np
import strax
import straxen
from straxen.common import get_resource, rotate_perp_wires
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO

export, __all__ = strax.exporter()


@export
class CorrectedAreas(strax.Plugin):
    """
    Plugin which applies light collection efficiency maps and electron
    life time to the data.
    Computes the cS1/cS2 for the main/alternative S1/S2 as well as the
    corrected life time.
    Note:
        Please be aware that for both, the main and alternative S1, the
        area is corrected according to the xy-position of the main S2.
        There are now 3 components of cS2s: cs2_top, cS2_bottom and cs2.
        cs2_top and cs2_bottom are corrected by the corresponding maps,
        and cs2 is the sum of the two.
    """
    __version__ = '0.3.1'

    depends_on = ['event_basics', 'event_positions']

    # Descriptor configs
    elife = straxen.URLConfig(
        default='cmt://elife?version=ONLINE&run_id=plugin.run_id',
        help='electron lifetime in [ns]')

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

    # cS2 AFT correction due to photon ionization
    cs2_bottom_top_ratio_correction = straxen.URLConfig(
        default=1,
        help='Scaling factor for cS2 AFT correction due to photon ionization'
    )

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for peak_type, peak_name in zip(['', 'alt_'], ['main', 'alternate']):
            dtype += [(f'{peak_type}cs1', np.float32, f'Corrected area of {peak_name} S1 [PE]'),
                      (f'{peak_type}cs1_wo_timecorr', np.float32,
                       f'Corrected area of {peak_name} S1 [PE] before time-dep LY correction'),
                      (f'{peak_type}cs2_wo_elifecorr', np.float32,
                       f'Corrected area of {peak_name} S2 before elife correction '
                       f'(s2 xy correction + SEG/EE correction applied) [PE]'),
                      (f'{peak_type}cs2_wo_timecorr', np.float32,
                       f'Corrected area of {peak_name} S2 before SEG/EE and elife corrections'
                       f'(s2 xy correction applied) [PE]'),
                      (f'{peak_type}cs2_area_fraction_top', np.float32,
                       f'Fraction of area seen by the top PMT array for corrected {peak_name} S2'),
                      (f'{peak_type}cs2_bottom', np.float32,
                       f'Corrected area of {peak_name} S2 in the bottom PMT array [PE]'),
                      (f'{peak_type}cs2', np.float32, f'Corrected area of {peak_name} S2 [PE]'), ]
        return dtype

    def ab_region(self, x, y):
        new_x, new_y = rotate_perp_wires(x, y)
        cond = new_x < self.region_linear
        cond &= new_x > -self.region_linear
        cond &= new_x ** 2 + new_y ** 2 < self.region_circular ** 2
        return cond

    def cd_region(self, x, y):
        return ~self.ab_region(x, y)

    def s2_map_names(self):

        # S2 top and bottom are corrected separately, and cS2 total is the sum of the two
        # figure out the map name
        if len(self.s2_xy_map.map_names) > 1:
            s2_top_map_name = "map_top"
            s2_bottom_map_name = "map_bottom"
        else:
            s2_top_map_name = "map"
            s2_bottom_map_name = "map"

        return s2_top_map_name, s2_bottom_map_name

    def seg_ee_correction_preparation(self):
        """Get single electron gain and extraction efficiency options"""
        self.regions = {'ab': self.ab_region, 'cd': self.cd_region}

        # setup SEG and EE corrections
        # if they are dicts, we just leave them as is
        # if they are not, we assume they are floats and
        # create a dict with the same correction in each region
        if isinstance(self.se_gain, dict):
            seg = self.se_gain
        else:
            seg = {key: self.se_gain for key in self.regions}

        if isinstance(self.avg_se_gain, dict):
            avg_seg = self.avg_se_gain
        else:
            avg_seg = {key: self.avg_se_gain for key in self.regions}

        if isinstance(self.rel_extraction_eff, dict):
            ee = self.rel_extraction_eff
        else:
            ee = {key: self.rel_extraction_eff for key in self.regions}
        
        return seg, avg_seg, ee

    def compute(self, events):
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']

        # S1 corrections depend on the actual corrected event position.
        # We use this also for the alternate S1; for e.g. Kr this is
        # fine as the S1 correction varies slowly.
        event_positions = np.vstack([events['x'], events['y'], events['z']]).T

        for peak_type in ["", "alt_"]:
            result[f"{peak_type}cs1_wo_timecorr"] = events[f'{peak_type}s1_area'] / self.s1_xyz_map(event_positions)
            result[f"{peak_type}cs1"] = result[f"{peak_type}cs1_wo_timecorr"] / self.rel_light_yield

        # s2 corrections
        s2_top_map_name, s2_bottom_map_name = self.s2_map_names()
        seg, avg_seg, ee = self.seg_ee_correction_preparation()

        # now can start doing corrections
        for peak_type in ["", "alt_"]:
            # S2(x,y) corrections use the observed S2 positions
            s2_positions = np.vstack([events[f'{peak_type}s2_x'], events[f'{peak_type}s2_y']]).T

            # corrected s2 with s2 xy map only, i.e. no elife correction
            # this is for s2-only events which don't have drift time info
            cs2_top_xycorr = (events[f'{peak_type}s2_area']
                              * events[f'{peak_type}s2_area_fraction_top']
                              / self.s2_xy_map(s2_positions, map_name=s2_top_map_name))
            cs2_bottom_xycorr = (events[f'{peak_type}s2_area']
                                 * (1 - events[f'{peak_type}s2_area_fraction_top'])
                                 / self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name))

            # For electron lifetime corrections to the S2s,
            # use drift time computed using the main S1.
            el_string = peak_type + "s2_interaction_" if peak_type == "alt_" else peak_type
            elife_correction = np.exp(events[f'{el_string}drift_time'] / self.elife)
            result[f"{peak_type}cs2_wo_timecorr"] = (cs2_top_xycorr + cs2_bottom_xycorr) * elife_correction

            for partition, func in self.regions.items():
                # partitioned SE and EE
                partition_mask = func(events[f'{peak_type}s2_x'], events[f'{peak_type}s2_y'])

                # Correct for SEgain and extraction efficiency
                seg_ee_corr = seg[partition] / avg_seg[partition] * ee[partition]

                # note that these are already masked!
                cs2_top_wo_elifecorr = cs2_top_xycorr[partition_mask] / seg_ee_corr
                cs2_bottom_wo_elifecorr = cs2_bottom_xycorr[partition_mask] / seg_ee_corr
                result[f"{peak_type}cs2_wo_elifecorr"][partition_mask] = cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr

                # cs2aft doesn't need elife/time corrections as they cancel
                result[f"{peak_type}cs2_area_fraction_top"][partition_mask] = cs2_top_wo_elifecorr / (cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr)
                result[f"{peak_type}cs2"][partition_mask] = result[f"{peak_type}cs2_wo_elifecorr"][partition_mask] * elife_correction[partition_mask]
                result[f"{peak_type}cs2_bottom"][partition_mask] = cs2_bottom_wo_elifecorr * elife_correction[partition_mask]

        # Photon ionization intensity and cS2 AFT correction (see #1247)
        for peak_type in ["", "alt_"]:
            cs2_bottom = result[f"{peak_type}cs2_bottom"]
            cs2_top = result[f"{peak_type}cs2"] - cs2_bottom

            # Bottom top ratios
            bt = cs2_bottom / cs2_top
            bt_corrected = bt * self.cs2_bottom_top_ratio_correction

            # cS2 bottom should be corrected by photon ionization, but not cS2 top
            cs2_top_corrected = cs2_top
            cs2_bottom_corrected = cs2_top * bt_corrected
            cs2_aft_corrected = cs2_top_corrected / (cs2_top_corrected + cs2_bottom_corrected)

            # Overwrite result
            result[f"{peak_type}cs2_area_fraction_top"] = cs2_aft_corrected
            result[f"{peak_type}cs2"] = cs2_top_corrected + cs2_bottom_corrected
            result[f"{peak_type}cs2_bottom"] = cs2_bottom_corrected

        return result
