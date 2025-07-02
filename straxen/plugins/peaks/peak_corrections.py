from straxen.plugins.events.corrected_areas import CorrectedAreas
import strax
import numpy as np
import straxen

export, __all__ = strax.exporter()


@export
class PeakCorrectedAreas(CorrectedAreas):
    """Pluging to apply corrections on peak level assuming that the main S1 is the only physical S1.

    We derived the average position of the S2s for the S1 LCE correction of S1s. We also correct the
    S2s for their individual (x,y) and drift time positions.

    """

    __version__ = "0.1.0"

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

    z_bias_map = straxen.URLConfig(
        default="itp_map://resource://XnT_z_bias_map_chargeup_20230329.json.gz"
        "?fmt=json.gz&method=RegularGridInterpolator",
        infer_type=False,
        help="Map of Z bias due to non uniform drift velocity/field",
    )

    fdc_map = straxen.URLConfig(
        default="xedocs://fdc_maps"
        "?algorithm=plugin.default_reconstruction_algorithm&run_id=plugin.run_id"
        "&attr=map&scale_coordinates=plugin.coordinate_scale&version=ONLINE",
        infer_type=False,
        help="3D field distortion correction map path",
    )

    def setup(self):
        """Setup the coordinate scales and FDC map."""
        self.coordinate_scales = [1.0, 1.0, -self.electron_drift_velocity]
        self.map = self.fdc_map

    def infer_dtype(self):
        dtype = strax.time_fields + [
            # S1 corrections
            ("cs1", np.float32, "Corrected S1 area [PE]"),
            (
                "cs1_wo_peakbiascorr",
                np.float32,
                "Corrected S1 area without peak bias correction [PE]",
            ),
            (
                "cs1_wo_xyzcorr",
                np.float32,
                "Corrected S1 area without xyz position correction [PE]",
            ),
            ("cs1_wo_timecorr", np.float32, "Corrected S1 area without time correction [PE]"),
            # S2 position corrections
            ("x_fdc", np.float32, "Field-distortion corrected x-position (cm)"),
            ("y_fdc", np.float32, "Field-distortion corrected y-position (cm)"),
            ("r_fdc", np.float32, "Field-distortion corrected r-position (cm)"),
            ("r_naive", np.float32, "Uncorrected r-position (cm)"),
            (
                "r_field_distortion_correction",
                np.float32,
                "Correction added to r_naive for field distortion (cm)",
            ),
            # S2 main corrections
            ("cs2", np.float32, "Corrected area of S2 [PE]"),
            ("cs2_bottom", np.float32, "Corrected area of S2 in the bottom PMT array [PE]"),
            (
                "cs2_area_fraction_top",
                np.float32,
                "Fraction of area seen by the top PMT array for corrected S2",
            ),
            # S2 intermediate corrections - for studying correction order
            (
                "cs2_wo_segee_picorr",
                np.float32,
                "Corrected S2 (before SEG/EE + photoionization, after peak bias + S2 xy + elife) [PE]",
            ),
            (
                "cs2_area_fraction_top_wo_segee_picorr",
                np.float32,
                "AFT for S2 (before SEG/EE + photoionization, after peak bias + S2 xy + elife)",
            ),
            (
                "cs2_before_pi",
                np.float32,
                "Corrected S2 (before photoionization + elife, after peak bias + S2 xy + SEG/EE) [PE]",
            ),
            (
                "cs2_area_fraction_top_before_pi",
                np.float32,
                "AFT for S2 (before photoionization + elife, after peak bias + S2 xy + SEG/EE)",
            ),
            (
                "cs2_before_elife",
                np.float32,
                "Corrected S2 (before elife, after peak bias + S2 xy + SEG/EE + photoionization) [PE]",
            ),
            (
                "cs2_area_fraction_top_before_elife",
                np.float32,
                "AFT for S2 (before elife, after peak bias + S2 xy + SEG/EE + photoionization)",
            ),
            # S2 N-1 corrections
            (
                "cs2_wo_timecorr",
                np.float32,
                "Corrected S2 (without time-dependent corrections) [PE]",
            ),
            (
                "cs2_area_fraction_top_wo_timecorr",
                np.float32,
                "AFT for S2 (without time-dependent corrections)",
            ),
            # Additional fields
            (
                "s1_xyz_correction_factor",
                np.float32,
                "Correction factor for the S1 area based on position",
            ),
            (
                "s1_rel_light_yield_correction_factor",
                np.float32,
                "Relative light yield correction factor for S1",
            ),
            ("z_obs_ms", np.float32, "z position of the multiscatter peak"),
        ]
        return dtype

    def compute(self, peaks):
        result = np.zeros(len(peaks), self.dtype)
        result["time"] = peaks["time"]
        result["endtime"] = peaks["endtime"]

        # Get z position of the peak with proper FDC
        z_obs = -self.electron_drift_velocity * peaks["drift_time"]
        z_obs = z_obs + self.electron_drift_velocity * self.electron_drift_time_gate

        # Apply z bias correction from FDC
        r_obs = np.hypot(peaks["x"], peaks["y"])
        z_dv_delta = self.z_bias_map(np.array([r_obs, z_obs]).T, map_name="z_bias_map")
        z_corr = z_obs - z_dv_delta

        result["z_obs_ms"] = z_corr

        # S1 correction factors
        s1_mask = peaks["type"] == 1
        s2_mask = peaks["type"] == 2

        # S1 corrections
        if np.any(s1_mask):
            # Get correction factors
            s1_area = peaks[s1_mask]["area"]
            s1_bias_corr = 1 + self.s1_bias_map(s1_area.reshape(-1, 1)).flatten()

            # Calculate z positions for xyz correction with proper FDC
            z_obs_s1 = -self.electron_drift_velocity * peaks[s1_mask]["drift_time"]
            z_obs_s1 = z_obs_s1 + self.electron_drift_velocity * self.electron_drift_time_gate

            # Apply z bias correction from FDC
            r_obs_s1 = np.hypot(peaks[s1_mask]["x"], peaks[s1_mask]["y"])
            z_dv_delta_s1 = self.z_bias_map(np.array([r_obs_s1, z_obs_s1]).T, map_name="z_bias_map")
            z_corr_s1 = z_obs_s1 - z_dv_delta_s1

            # Use corrected z for S1 position
            s1_positions = np.vstack([peaks[s1_mask]["x"], peaks[s1_mask]["y"], z_corr_s1]).T

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

            # Apply field distortion correction to S2 positions
            x_s2 = peaks[s2_mask]["x"]
            y_s2 = peaks[s2_mask]["y"]
            z_s2 = -self.electron_drift_velocity * peaks[s2_mask]["drift_time"]
            z_s2 = z_s2 + self.electron_drift_velocity * self.electron_drift_time_gate

            # Calculate r_obs and apply z bias correction
            r_obs_s2 = np.hypot(x_s2, y_s2)
            z_dv_delta_s2 = self.z_bias_map(np.array([r_obs_s2, z_s2]).T, map_name="z_bias_map")
            z_corr_s2 = z_s2 - z_dv_delta_s2

            # Create corrected positions for FDC map
            corr_pos_s2 = np.vstack([x_s2, y_s2, z_corr_s2]).T

            # Apply r bias correction using FDC map
            delta_r_s2 = self.map(corr_pos_s2)
            r_cor_s2 = r_obs_s2 + delta_r_s2

            # Scale x,y using ratio of corrected r / original r
            with np.errstate(invalid="ignore", divide="ignore"):
                scale_s2 = np.divide(
                    r_cor_s2, r_obs_s2, out=np.zeros_like(r_cor_s2), where=r_obs_s2 != 0
                )

            # Store FDC corrected positions in result array
            result["x_fdc"][s2_mask] = x_s2 * scale_s2
            result["y_fdc"][s2_mask] = y_s2 * scale_s2
            result["r_fdc"][s2_mask] = r_cor_s2
            result["r_naive"][s2_mask] = r_obs_s2
            result["r_field_distortion_correction"][s2_mask] = delta_r_s2

            # Use original observed positions for S2 xy maps in gas gap
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
            result["cs2_bottom"][s2_mask] = result["cs2"][s2_mask] * (
                1 - result["cs2_area_fraction_top"][s2_mask]
            )

            # We only keep the wo_timecorr variant for S2s

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
            )[
                :2
            ]

        return result
