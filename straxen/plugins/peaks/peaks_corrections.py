from straxen.plugins.events.corrected_areas import CorrectedAreas
import strax
import numpy as np
import straxen

export, __all__ = strax.exporter()


@export
class PeakCorrectedAreas(CorrectedAreas):
    __version__ = '0.0.0'

    depends_on = ['peak_basics', 'peak_positions', 'peaks_per_event']
    data_kind = 'peaks'
    provides = 'peaks_corrections'

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    electron_drift_time_gate = straxen.URLConfig(
        default='cmt://'
                'electron_drift_time_gate'
                '?version=ONLINE&run_id=plugin.run_id',
        help='Electron drift time from the gate in ns',
        cache=True)

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields
        dtype += [(f'cs2_wo_elifecorr', np.float32,
                   f'Corrected area of S2 before elife correction '
                   f'(s2 xy correction + SEG/EE correction applied) [PE]'),
                  (f'cs2_wo_timecorr', np.float32,
                   f'Corrected area of S2 before SEG/EE and elife corrections'
                   f'(s2 xy correction applied) [PE]'),
                  (f'cs2_area_fraction_top', np.float32,
                   f'Fraction of area seen by the top PMT array for corrected  S2'),
                  (f'cs2_bottom', np.float32,
                   f'Corrected area of S2 in the bottom PMT array [PE]'),
                  (f'cs2', np.float32, f'Corrected area of  S2 [PE]'),
                  (f's1_xyz_correction_factor', np.float32,
                   f'Correction factor for the S1 area based on S2 position'),
                  (f's1_rel_light_yield_correction_factor', np.float32,
                   f'Relative light yield correction factor for the S1 area'),
                  ]
        return dtype

    def compute(self, peaks):
        result = np.zeros(len(peaks), self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = peaks['endtime']

        # Get S1 correction factors
        z_obs = - self.electron_drift_velocity * peaks[f'drift_time']
        z_obs = z_obs + self.electron_drift_velocity * self.electron_drift_time_gate
        peak_positions = np.vstack([peaks['x'], peaks['y'], z_obs]).T
        result["s1_xyz_correction_factor"] = 1 / self.s1_xyz_map(peak_positions)
        result["s1_rel_light_yield_correction_factor"] = 1 / self.rel_light_yield

        # s2 corrections
        s2_top_map_name, s2_bottom_map_name = self.s2_map_names()

        seg, avg_seg, ee = self.seg_ee_correction_preparation()

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
            seg_ee_corr = seg[partition] / avg_seg[partition] * ee[partition]

            # note that these are already masked!
            cs2_top_wo_elifecorr = cs2_top_xycorr[partition_mask] / seg_ee_corr
            cs2_bottom_wo_elifecorr = cs2_bottom_xycorr[partition_mask] / seg_ee_corr

            result[f"cs2_wo_elifecorr"][partition_mask] = cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr

            # cs2aft doesn't need elife/time corrections as they cancel
            result[f"cs2_area_fraction_top"][partition_mask] = cs2_top_wo_elifecorr / (
                        cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr)

            result[f"cs2"][partition_mask] = result[f"cs2_wo_elifecorr"][partition_mask] * elife_correction[
                partition_mask]
            result[f"cs2_bottom"][partition_mask] = cs2_bottom_wo_elifecorr * elife_correction[partition_mask]
        result[f"cs2_wo_timecorr"][peaks["type"] != 2] = np.nan
        result[f"cs2_wo_elifecorr"][peaks["type"] != 2] = np.nan
        result[f"cs2_area_fraction_top"][peaks["type"] != 2] = np.nan
        result[f"cs2"][peaks["type"] != 2] = np.nan
        result[f"cs2_bottom"][peaks["type"] != 2] = np.nan
        result["s1_xyz_correction_factor"][peaks["type"] != 2] = np.nan
        result["s1_rel_light_yield_correction_factor"][peaks["type"] != 2] = np.nan
        return result
