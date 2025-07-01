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
            ("cs1_wo_xyzcorr", np.float32, "Bias-corrected S1 area before xyz correction [PE]"),
            ("cs2_wo_xycorr", np.float32, "Bias-corrected S2 area before xy correction [PE]"),
            (
                "cs2_wo_elifecorr",
                np.float32,
                "Corrected S2 area before elife correction (s2 xy, SEG/EE applied) [PE]",
            ),
            (
                "cs2_wo_timecorr",
                np.float32,
                "Corrected S2 area before SEG/EE and elife (s2 xy applied) [PE]",
            ),
            (
                "cs2_area_fraction_top",
                np.float32,
                "Fraction of area seen by the top PMT array for corrected S2",
            ),
            ("cs2_bottom", np.float32, "Corrected area of S2 in the bottom PMT array [PE]"),
            ("cs2", np.float32, "Corrected area of S2 [PE]"),
            (
                "s1_xyz_correction_factor",
                np.float32,
                "Correction factor for the S1 area based on S2 position",
            ),
            (
                "s1_rel_light_yield_correction_factor",
                np.float32,
                "Relative light yield correction factor for the S1 area",
            ),
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
            # Bias correction
            s1_bias_corr = 1 + self.s1_bias_map(peaks[s1_mask]["area"].reshape(-1, 1)).flatten()
            result["cs1_wo_xyzcorr"][s1_mask] = peaks[s1_mask]["area"] / s1_bias_corr

            # LCE and time corrections (as factors)
            z_obs = -self.electron_drift_velocity * peaks["drift_time"]
            z_obs = z_obs + self.electron_drift_velocity * self.electron_drift_time_gate
            s1_positions = np.vstack([peaks[s1_mask]["x"], peaks[s1_mask]["y"], z_obs[s1_mask]]).T

            result["s1_xyz_correction_factor"][s1_mask] = 1 / self.s1_xyz_map(s1_positions)
            result["s1_rel_light_yield_correction_factor"][s1_mask] = 1 / self.rel_light_yield

        # S2 corrections
        if np.any(s2_mask):
            # --- Start sequential corrections ---
            # 1. Bias correction
            s2_bias_corr = 1 + self.s2_bias_map(peaks[s2_mask]["area"].reshape(-1, 1)).flatten()
            cs2_after_bias = peaks[s2_mask]["area"] / s2_bias_corr
            result["cs2_wo_xycorr"][s2_mask] = cs2_after_bias

            # 2. S2 XY LCE correction
            s2_positions = np.vstack([peaks[s2_mask]["x"], peaks[s2_mask]["y"]]).T
            s2_top_map_name, s2_bottom_map_name = self.s2_map_names()
            s2_aft = peaks[s2_mask]["area_fraction_top"]

            cs2_top_after_xy = (
                cs2_after_bias
                * s2_aft
                / self.s2_xy_map(s2_positions, map_name=s2_top_map_name)
            )
            cs2_bottom_after_xy = (
                cs2_after_bias
                * (1 - s2_aft)
                / self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name)
            )
            cs2_after_xy = cs2_top_after_xy + cs2_bottom_after_xy
            result["cs2_wo_timecorr"][s2_mask] = cs2_after_xy

            # 3. SEG/EE correction
            seg, avg_seg, ee = self.seg_ee_correction_preparation()
            seg_ee_corr = np.ones_like(cs2_after_xy)
            for partition, func in self.regions.items():
                partition_mask = func(peaks[s2_mask]["x"], peaks[s2_mask]["y"])
                seg_ee_corr[partition_mask] = seg[partition] / avg_seg[partition] * ee[partition]

            cs2_top_after_segee = cs2_top_after_xy / seg_ee_corr
            cs2_bottom_after_segee = cs2_bottom_after_xy / seg_ee_corr

            # 3b. Photoionization correction for S2 bottom
            cs2_bottom_after_segee *= self.cs2_bottom_top_ratio_correction

            cs2_after_segee = cs2_top_after_segee + cs2_bottom_after_segee
            result["cs2_wo_elifecorr"][s2_mask] = cs2_after_segee

            # 4. Electron lifetime correction
            elife_correction = np.exp(peaks[s2_mask]["drift_time"] / self.elife)
            cs2_final = cs2_after_segee * elife_correction
            result["cs2"][s2_mask] = cs2_final

            # --- Final derived quantities ---
            cs2_top_final = cs2_top_after_segee * elife_correction
            cs2_bottom_final = cs2_bottom_after_segee * elife_correction
            result["cs2_bottom"][s2_mask] = cs2_bottom_final
            with np.errstate(invalid="ignore", divide="ignore"):
                result["cs2_area_fraction_top"][s2_mask] = cs2_top_final / cs2_final

            # Z position for S2s
            z_obs = -self.electron_drift_velocity * peaks[s2_mask]["drift_time"]
            z_obs = z_obs + self.electron_drift_velocity * self.electron_drift_time_gate
            result["z_obs_ms"][s2_mask] = z_obs

        return result
