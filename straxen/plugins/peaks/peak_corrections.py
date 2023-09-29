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
        default="cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    electron_drift_time_gate = straxen.URLConfig(
        default="cmt://electron_drift_time_gate?version=ONLINE&run_id=plugin.run_id",
        help="Electron drift time from the gate in ns",
        cache=True,
    )

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (
                (
                    "Corrected area of S2 before elife correction "
                    "(s2 xy correction + SEG/EE correction applied) [PE]",
                    "cs2_wo_elifecorr",
                ),
                np.float32,
            ),
            (
                (
                    "Corrected area of S2 before SEG/EE and elife corrections "
                    "(s2 xy correction applied) [PE]",
                    "cs2_wo_timecorr",
                ),
                np.float32,
            ),
            (
                (
                    "Fraction of area seen by the top PMT array for corrected S2",
                    "cs2_area_fraction_top",
                ),
                np.float32,
            ),
            (("Corrected area of S2 in the bottom PMT array [PE]", "cs2_bottom"), np.float32),
            (("Corrected area of S2 [PE]", "cs2"), np.float32),
            (
                (
                    "Correction factor for the S1 area based on S2 position",
                    "s1_xyz_correction_factor",
                ),
                np.float32,
            ),
            (
                (
                    "Relative light yield correction factor for the S1 area",
                    "s1_rel_light_yield_correction_factor",
                ),
                np.float32,
            ),
            (("z position of the multiscatter peak", "z_obs_ms"), np.float32),
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

        # Get S1 correction factors
        peak_positions = np.vstack([peaks["x"], peaks["y"], z_obs]).T
        result["s1_xyz_correction_factor"] = 1 / self.s1_xyz_map(peak_positions)
        result["s1_rel_light_yield_correction_factor"] = 1 / self.rel_light_yield

        # s2 corrections
        s2_top_map_name, s2_bottom_map_name = self.s2_map_names()

        seg, avg_seg, ee = self.seg_ee_correction_preparation()

        # now can start doing corrections

        # S2(x,y) corrections use the observed S2 positions
        s2_positions = np.vstack([peaks["x"], peaks["y"]]).T

        # corrected s2 with s2 xy map only, i.e. no elife correction
        # this is for s2-only events which don't have drift time info

        cs2_top_xycorr = (
            peaks["area"]
            * peaks["area_fraction_top"]
            / self.s2_xy_map(s2_positions, map_name=s2_top_map_name)
        )
        cs2_bottom_xycorr = (
            peaks["area"]
            * (1 - peaks["area_fraction_top"])
            / self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name)
        )

        # For electron lifetime corrections to the S2s,
        # use drift time computed using the main S1.

        elife_correction = np.exp(peaks["drift_time"] / self.elife)
        result["cs2_wo_timecorr"] = (cs2_top_xycorr + cs2_bottom_xycorr) * elife_correction

        for partition, func in self.regions.items():
            # partitioned SE and EE
            partition_mask = func(peaks["x"], peaks["y"])

            # Correct for SEgain and extraction efficiency
            seg_ee_corr = seg[partition] / avg_seg[partition] * ee[partition]

            # note that these are already masked!
            cs2_top_wo_elifecorr = cs2_top_xycorr[partition_mask] / seg_ee_corr
            cs2_bottom_wo_elifecorr = cs2_bottom_xycorr[partition_mask] / seg_ee_corr

            result["cs2_wo_elifecorr"][partition_mask] = (
                cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr
            )

            # cs2aft doesn't need elife/time corrections as they cancel
            result["cs2_area_fraction_top"][partition_mask] = cs2_top_wo_elifecorr / (
                cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr
            )

            result["cs2"][partition_mask] = (
                result["cs2_wo_elifecorr"][partition_mask] * elife_correction[partition_mask]
            )
            result["cs2_bottom"][partition_mask] = (
                cs2_bottom_wo_elifecorr * elife_correction[partition_mask]
            )

        not_s2_mask = peaks["type"] != 2
        result["cs2_wo_timecorr"][not_s2_mask] = np.nan
        result["cs2_wo_elifecorr"][not_s2_mask] = np.nan
        result["cs2_area_fraction_top"][not_s2_mask] = np.nan
        result["cs2"][not_s2_mask] = np.nan
        result["z_obs_ms"][not_s2_mask] = np.nan
        result["cs2_bottom"][not_s2_mask] = np.nan
        result["s1_xyz_correction_factor"][not_s2_mask] = np.nan
        result["s1_rel_light_yield_correction_factor"][not_s2_mask] = np.nan
        return result
