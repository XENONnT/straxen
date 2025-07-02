from straxen.plugins.events.corrected_areas import CorrectedAreas
import strax
import numpy as np
import straxen

export, __all__ = strax.exporter()


@export
class PeakCorrectedAreas(CorrectedAreas):
    """Pluging to apply corrections on peak level assuming that the main S1 is the only physical
    S1."""

    __version__ = "0.0.1"

    depends_on = ("peak_basics", "peak_positions", "peak_per_event")
    data_kind = "peaks"
    provides = "peak_corrections"

    electron_drift_velocity = straxen.URLConfig(
        default="xedocs://electron_drift_velocities?attr=value&run_id=plugin.run_id&version=ONLINE",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    electron_drift_time_gate = straxen.URLConfig(
        default="xedocs://electron_drift_time_gates?attr=value&run_id=plugin.run_id&version=ONLINE",
        help="Electron drift time from the gate in ns",
        cache=True,
    )

    def infer_dtype(self):
        dtype = strax.time_fields + [
            # S1 corrections
            ("cs1", np.float32, "Corrected S1 area [PE]"),
            ("cs1_wo_peakbiascorr", np.float32, "Corrected S1 area without peak bias correction [PE]"),
            ("cs1_wo_xyzcorr", np.float32, "Corrected S1 area without xyz position correction [PE]"),
            ("cs1_wo_timecorr", np.float32, "Corrected S1 area without time correction [PE]"),
            
            # S2 main corrections
            ("cs2", np.float32, "Corrected area of S2 [PE]"),
            ("cs2_bottom", np.float32, "Corrected area of S2 in the bottom PMT array [PE]"),
            ("cs2_area_fraction_top", np.float32, "Fraction of area seen by the top PMT array for corrected S2"),
            
            # S2 intermediate corrections
            ("cs2_before_pi", np.float32, "Corrected S2 before photoionization correction [PE]"),
            ("cs2_area_fraction_top_before_pi", np.float32, "AFT for S2 before photoionization correction"),
            
            ("cs2_before_elife", np.float32, "Corrected S2 before electron lifetime correction [PE]"),
            ("cs2_area_fraction_top_before_elife", np.float32, "AFT for S2 before electron lifetime correction"),
            
            ("cs2_wo_segee_picorr", np.float32, "Corrected S2 without SEG/EE and photoionization corrections [PE]"),
            ("cs2_area_fraction_top_wo_segee_picorr", np.float32, "AFT for S2 without SEG/EE and photoionization"),
            
            # S2 N-1 corrections
            ("cs2_wo_peakbiascorr", np.float32, "Corrected S2 without peak bias correction [PE]"),
            ("cs2_area_fraction_top_wo_peakbiascorr", np.float32, "AFT for S2 without peak bias correction"),
            
            ("cs2_wo_xycorr", np.float32, "Corrected S2 without xy position correction [PE]"),
            ("cs2_area_fraction_top_wo_xycorr", np.float32, "AFT for S2 without xy position correction"),
            
            ("cs2_wo_segee", np.float32, "Corrected S2 without SEG/EE correction [PE]"),
            ("cs2_area_fraction_top_wo_segee", np.float32, "AFT for S2 without SEG/EE correction"),
            
            ("cs2_wo_picorr", np.float32, "Corrected S2 without photoionization correction [PE]"),
            ("cs2_area_fraction_top_wo_picorr", np.float32, "AFT for S2 without photoionization correction"),
            
            ("cs2_wo_elifecorr", np.float32, "Corrected S2 without electron lifetime correction [PE]"),
            ("cs2_area_fraction_top_wo_elifecorr", np.float32, "AFT for S2 without electron lifetime correction"),
            
            ("cs2_wo_timecorr", np.float32, "Corrected S2 without any time-dependent corrections [PE]"),
            ("cs2_area_fraction_top_wo_timecorr", np.float32, "AFT for S2 without any time-dependent corrections"),
            
            # Additional fields
            ("s1_xyz_correction_factor", np.float32, "Correction factor for the S1 area based on position"),
            ("s1_rel_light_yield_correction_factor", np.float32, "Relative light yield correction factor for S1"),
            ("z_obs_ms", np.float32, "z position of the multiscatter peak"),
        ]
        return dtype

    def compute(self, peaks):
        result = np.zeros(len(peaks), self.dtype)
        result["time"] = peaks["time"]
        result["endtime"] = peaks["endtime"]

        # Get z position of the peak
        z_obs = -self.electron_drift_velocity * peaks["drift_time"]
        z_obs = z_obs + self.electron_drift_velocity * self.electron_drift_time_gate
        result["z_obs_ms"] = z_obs

        # S1 correction factors
        s1_mask = peaks["type"] == 1
        s2_mask = peaks["type"] == 2

        # S1 corrections
        if np.any(s1_mask):
            # Get correction factors
            s1_area = peaks[s1_mask]["area"]
            s1_bias_corr = 1 + self.s1_bias_map(s1_area.reshape(-1, 1)).flatten()
            
            # Calculate z positions for xyz correction
            z_obs_s1 = -self.electron_drift_velocity * peaks[s1_mask]["drift_time"]
            z_obs_s1 = z_obs_s1 + self.electron_drift_velocity * self.electron_drift_time_gate
            s1_positions = np.vstack([peaks[s1_mask]["x"], peaks[s1_mask]["y"], z_obs_s1]).T
            
            s1_xyz_corr = self.s1_xyz_map(s1_positions)
            s1_time_corr = self.rel_light_yield
            
            # Store correction factors
            result["s1_xyz_correction_factor"][s1_mask] = 1 / s1_xyz_corr
            result["s1_rel_light_yield_correction_factor"][s1_mask] = 1 / s1_time_corr
            
            # Apply all corrections
            result["cs1"][s1_mask] = s1_area / s1_bias_corr / s1_xyz_corr / s1_time_corr
            
            # N-1 corrections for S1
            result["cs1_wo_peakbiascorr"][s1_mask] = s1_area / s1_xyz_corr / s1_time_corr
            result["cs1_wo_xyzcorr"][s1_mask] = s1_area / s1_bias_corr / s1_time_corr
            result["cs1_wo_timecorr"][s1_mask] = s1_area / s1_bias_corr / s1_xyz_corr

        # S2 corrections
        if np.any(s2_mask):
            # 0. Get all correction factors
            s2_area = peaks[s2_mask]["area"]
            s2_aft = peaks[s2_mask]["area_fraction_top"]
            s2_positions = np.vstack([peaks[s2_mask]["x"], peaks[s2_mask]["y"]]).T
            s2_top_map_name, s2_bottom_map_name = self.s2_map_names()

            s2_bias_corr = 1 + self.s2_bias_map(s2_area.reshape(-1, 1)).flatten()
            s2_xy_corr_top = self.s2_xy_map(s2_positions, map_name=s2_top_map_name)
            s2_xy_corr_bottom = self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name)
            
            seg, avg_seg, ee = self.seg_ee_correction_preparation()
            seg_ee_corr = np.ones_like(peaks[s2_mask]["x"], dtype=np.float32)
            for partition, func in self.regions.items():
                partition_mask = func(peaks[s2_mask]["x"], peaks[s2_mask]["y"])
                seg_ee_corr[partition_mask] = seg[partition] / avg_seg[partition] * ee[partition]
            
            pi_corr_bottom = self.cs2_bottom_top_ratio_correction
            elife_correction = np.exp(peaks[s2_mask]["drift_time"] / self.elife)

            # 1. Apply all corrections directly to result array
            # Full corrections
            (
                result["cs2"][s2_mask],
                result["cs2_area_fraction_top"][s2_mask],
                result["cs2_before_pi"][s2_mask],
                result["cs2_area_fraction_top_before_pi"][s2_mask],
                result["cs2_before_elife"][s2_mask],
                result["cs2_area_fraction_top_before_elife"][s2_mask],
                result["cs2_wo_segee_picorr"][s2_mask],
                result["cs2_area_fraction_top_wo_segee_picorr"][s2_mask],
            ) = self.apply_s2_corrections(
                s2_area=s2_area,
                s2_aft=s2_aft,
                s2_bias_correction=s2_bias_corr,
                s2_xy_correction_top=s2_xy_corr_top,
                s2_xy_correction_bottom=s2_xy_corr_bottom,
                seg_ee_corr=seg_ee_corr,
                pi_corr_bottom=pi_corr_bottom,
                elife_correction=elife_correction,
            )
            
            # Calculate cs2_bottom directly
            result["cs2_bottom"][s2_mask] = result["cs2"][s2_mask] * (1 - result["cs2_area_fraction_top"][s2_mask])

            # N-1 corrections - without peak bias
            (
                result["cs2_wo_peakbiascorr"][s2_mask],
                result["cs2_area_fraction_top_wo_peakbiascorr"][s2_mask],
            ) = self.apply_s2_corrections(
                s2_area=s2_area,
                s2_aft=s2_aft,
                s2_bias_correction=1,  # Without peak bias
                s2_xy_correction_top=s2_xy_corr_top,
                s2_xy_correction_bottom=s2_xy_corr_bottom,
                seg_ee_corr=seg_ee_corr,
                pi_corr_bottom=pi_corr_bottom,
                elife_correction=elife_correction,
            )[:2]

            # N-1 corrections - without xy correction
            (
                result["cs2_wo_xycorr"][s2_mask],
                result["cs2_area_fraction_top_wo_xycorr"][s2_mask],
            ) = self.apply_s2_corrections(
                s2_area=s2_area,
                s2_aft=s2_aft,
                s2_bias_correction=s2_bias_corr,
                s2_xy_correction_top=1,  # Without xy correction
                s2_xy_correction_bottom=1,  # Without xy correction
                seg_ee_corr=seg_ee_corr,
                pi_corr_bottom=pi_corr_bottom,
                elife_correction=elife_correction,
            )[:2]

            # N-1 corrections - without SEG/EE
            (
                result["cs2_wo_segee"][s2_mask],
                result["cs2_area_fraction_top_wo_segee"][s2_mask],
            ) = self.apply_s2_corrections(
                s2_area=s2_area,
                s2_aft=s2_aft,
                s2_bias_correction=s2_bias_corr,
                s2_xy_correction_top=s2_xy_corr_top,
                s2_xy_correction_bottom=s2_xy_corr_bottom,
                seg_ee_corr=1,  # Without SEG/EE
                pi_corr_bottom=pi_corr_bottom,
                elife_correction=elife_correction,
            )[:2]

            # N-1 corrections - without photoionization
            (
                result["cs2_wo_picorr"][s2_mask],
                result["cs2_area_fraction_top_wo_picorr"][s2_mask],
            ) = self.apply_s2_corrections(
                s2_area=s2_area,
                s2_aft=s2_aft,
                s2_bias_correction=s2_bias_corr,
                s2_xy_correction_top=s2_xy_corr_top,
                s2_xy_correction_bottom=s2_xy_corr_bottom,
                seg_ee_corr=seg_ee_corr,
                pi_corr_bottom=1,  # Without photoionization
                elife_correction=elife_correction,
            )[:2]

            # N-1 corrections - without electron lifetime
            (
                result["cs2_wo_elifecorr"][s2_mask],
                result["cs2_area_fraction_top_wo_elifecorr"][s2_mask],
            ) = self.apply_s2_corrections(
                s2_area=s2_area,
                s2_aft=s2_aft,
                s2_bias_correction=s2_bias_corr,
                s2_xy_correction_top=s2_xy_corr_top,
                s2_xy_correction_bottom=s2_xy_corr_bottom,
                seg_ee_corr=seg_ee_corr,
                pi_corr_bottom=pi_corr_bottom,
                elife_correction=1,  # Without electron lifetime
            )[:2]

            # N-1 corrections - without time dependent corrections
            (
                result["cs2_wo_timecorr"][s2_mask],
                result["cs2_area_fraction_top_wo_timecorr"][s2_mask],
            ) = self.apply_s2_corrections(
                s2_area=s2_area,
                s2_aft=s2_aft,
                s2_bias_correction=s2_bias_corr,
                s2_xy_correction_top=s2_xy_corr_top,
                s2_xy_correction_bottom=s2_xy_corr_bottom,
                seg_ee_corr=1,  # No SEG/EE (time dependent)
                pi_corr_bottom=pi_corr_bottom,
                elife_correction=1,  # No electron lifetime (time dependent)
            )[:2]

        return result
