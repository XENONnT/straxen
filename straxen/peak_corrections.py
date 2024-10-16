import numpy as np
import straxen

# correction_utils.py
import numpy as np
import straxen
from straxen.common import rotate_perp_wires

def ab_region(x, y, single_electron_gain_partition):
    """Return True if (x, y) is in the AB region based on the partition."""
    new_x, new_y = rotate_perp_wires(x, y)
    cond = new_x < single_electron_gain_partition["linear"]
    cond &= new_x > -single_electron_gain_partition["linear"]
    cond &= new_x**2 + new_y**2 < single_electron_gain_partition["circular"] ** 2
    return cond

def cd_region(x, y, single_electron_gain_partition):
    """Return True if (x, y) is in the CD region."""
    return ~ab_region(x, y, single_electron_gain_partition)

def s2_map_names(s2_xy_map):
    """Return map names for S2 top and bottom corrections."""
    if len(s2_xy_map.map_names) > 1:
        s2_top_map_name = "map_top"
        s2_bottom_map_name = "map_bottom"
    else:
        s2_top_map_name = "map"
        s2_bottom_map_name = "map"
    return s2_top_map_name, s2_bottom_map_name

def seg_ee_correction_preparation(self):
    """Get single electron gain and extraction efficiency options."""
    self.regions = {"ab": ab_region, "cd": cd_region}

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

def apply_s1_corrections(self, events):
    """Apply corrections to S1 signals."""
    event_positions = np.vstack([events["x"], events["y"], events["z"]]).T
    result = {}

    for peak_type in ["", "alt_"]:
        result[f"{peak_type}cs1_wo_timecorr"] = events[f"{peak_type}s1_area"] / self.s1_xyz_map(
            event_positions
        )
        result[f"{peak_type}cs1"] = result[f"{peak_type}cs1_wo_timecorr"] / self.rel_light_yield

    return result

def apply_s2_corrections(self, events, seg, avg_seg, ee):
    """Apply corrections to S2 signals."""
    result = {}
    regions = {"ab": ab_region, "cd": cd_region}
    s2_top_map_name, s2_bottom_map_name = s2_map_names(self.s2_xy_map)
    
    for peak_type in ["", "alt_"]:
        s2_positions = np.vstack([events[f"{peak_type}s2_x"], events[f"{peak_type}s2_y"]]).T


        # corrected S2 with S2(x,y) map only, i.e. no elife correction
        # this is for S2-only events which don't have drift time info
        s2_xy_top = self.s2_xy_map(s2_positions, map_name=s2_top_map_name)
        cs2_top_xycorr = (
            events[f"{peak_type}s2_area"]
            * events[f"{peak_type}s2_area_fraction_top"]
            / s2_xy_top
        )
        s2_xy_bottom = self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name)
        cs2_bottom_xycorr = (
            events[f"{peak_type}s2_area"]
            * (1 - events[f"{peak_type}s2_area_fraction_top"])
            / s2_xy_bottom
        )

        # use drift time computed using the main S1.
        el_string = peak_type + "s2_interaction_" if peak_type == "alt_" else peak_type
        elife_correction = np.exp(events[f"{el_string}drift_time"] / self.elife)
        
        # Apply SEG and EE corrections
        seg_ee_corr = np.zeros(len(events))
        for partition, func in regions.items():
            partition_mask = func(events[f"{peak_type}s2_x"], events[f"{peak_type}s2_y"], self.single_electron_gain_partition)
            seg_ee_corr[partition_mask] = seg[partition] / avg_seg[partition] * ee[partition]

        cs2_xycorr = cs2_top_xycorr + cs2_bottom_xycorr
        result[f"{peak_type}cs2_wo_timecorr"] = cs2_xycorr * elife_correction
        result[f"{peak_type}cs2"] = result[f"{peak_type}cs2_wo_timecorr"]

    return result

def apply_all_corrections(self, events):
    """Apply all corrections (S1, S2, etc.) in one function."""
    result = {}

    # Apply S1 corrections
    s1_result = apply_s1_corrections(self, events)
    result.update(s1_result)

    # Prepare SEG and EE corrections
    seg, avg_seg, ee = seg_ee_correction_preparation(self)

    # Apply S2 corrections
    s2_result = apply_s2_corrections(self, events, seg, avg_seg, ee)
    result.update(s2_result)

    return result

def infer_correction_dtype():
    """Return the dtype for the plugin output."""
    dtype = []
    dtype += strax.time_fields

    for peak_type, peak_name in zip(["", "alt_"], ["main", "alternate"]):
        dtype += [
            (f"{peak_type}cs1", np.float32, f"Corrected area of {peak_name} S1 [PE]"),
            (
                f"{peak_type}cs1_wo_timecorr",
                np.float32,
                f"Corrected area of {peak_name} S1 (before LY correction) [PE]",
            ),
        ]
        names = ["_wo_timecorr", "_wo_picorr", "_wo_elifecorr", ""]
        descriptions = ["S2 xy", "SEG/EE", "photon ionization", "elife"]
        for i, name in enumerate(names):
            if i == len(names) - 1:
                description = ""
            elif i == 0:
                # special treatment for wo_timecorr, apply elife correction
                description = " (before " + " + ".join(descriptions[i + 1 : -1])
                description += (
                    ", after " + " + ".join(descriptions[: i + 1] + descriptions[-1:]) + ")"
                )
            else:
                description = " (before " + " + ".join(descriptions[i + 1 :])
                description += ", after " + " + ".join(descriptions[: i + 1]) + ")"
            dtype += [
                (
                    f"{peak_type}cs2{name}",
                    np.float32,
                    f"Corrected area of {peak_name} S2{description} [PE]",
                ),
                (
                    f"{peak_type}cs2_area_fraction_top{name}",
                    np.float32,
                    (
                        "Fraction of area seen by the top PMT array for corrected "
                        f"{peak_name} S2{description}"
                    ),
                ),
            ]
    return dtype