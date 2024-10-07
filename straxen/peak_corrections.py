# peak_corrections.py

import numpy as np
import straxen

def apply_s1_corrections(event_positions, s1_area, s1_xyz_map, rel_light_yield):
    """
    Apply S1 corrections based on event position and other correction factors.

    Parameters:
    - event_positions: Array of (x, y, z) positions.
    - s1_area: Array of S1 area values.
    - s1_xyz_map: XYZ map for S1 correction.
    - rel_light_yield: Light yield correction factor.

    Returns:
    - corrected_s1: Corrected S1 areas.
    - corrected_s1_wo_timecorr: Corrected S1 areas without time-dependent corrections.
    """
    corrected_s1_wo_timecorr = s1_area / s1_xyz_map(event_positions)
    corrected_s1 = corrected_s1_wo_timecorr / rel_light_yield
    return corrected_s1, corrected_s1_wo_timecorr


def apply_s2_corrections(peaks, s2_positions, s2_xy_map, elife, seg, avg_seg, ee, cs2_bottom_top_ratio_correction):
    """
    Apply S2 corrections including XY map, elife correction, SEG/EE, and photon ionization correction.

    Parameters:
    - peaks: Array of peak data (including S2 areas and positions).
    - s2_positions: Array of (x, y) positions for S2 peaks.
    - s2_xy_map: XY map for S2 correction.
    - elife: Electron lifetime value.
    - seg: Single electron gain corrections.
    - avg_seg: Average SEG corrections.
    - ee: Extraction efficiency corrections.
    - cs2_bottom_top_ratio_correction: Correction factor for S2 bottom/top ratio.

    Returns:
    - Corrected S2 areas and fractions as a dictionary of arrays.
    """
    s2_top_map_name, s2_bottom_map_name = _determine_s2_map_names(s2_xy_map)

    # Apply XY correction to S2 peaks
    s2_xy_top = s2_xy_map(s2_positions, map_name=s2_top_map_name)
    cs2_top_xycorr = peaks['area'] * peaks['area_fraction_top'] / s2_xy_top
    s2_xy_bottom = s2_xy_map(s2_positions, map_name=s2_bottom_map_name)
    cs2_bottom_xycorr = peaks['area'] * (1 - peaks['area_fraction_top']) / s2_xy_bottom

    # Electron lifetime correction
    elife_correction = np.exp(peaks['drift_time'] / elife)

    # SEG and EE corrections
    seg_ee_corr = np.zeros(len(peaks))
    for partition, func in {"ab": _ab_region, "cd": _cd_region}.items():
        partition_mask = func(s2_positions[:, 0], s2_positions[:, 1])
        seg_ee_corr[partition_mask] = seg[partition] / avg_seg[partition] * ee[partition]

    # Combine corrections
    cs2_xycorr = cs2_top_xycorr + cs2_bottom_xycorr
    cs2_wo_timecorr = cs2_xycorr * elife_correction

    cs2_top_wo_picorr = cs2_top_xycorr / seg_ee_corr
    cs2_bottom_wo_picorr = cs2_bottom_xycorr / seg_ee_corr

    # Photon ionization correction
    cs2_bottom_wo_elifecorr = cs2_bottom_wo_picorr * cs2_bottom_top_ratio_correction
    cs2_top_wo_elifecorr = cs2_top_wo_picorr
    cs2_wo_elifecorr = cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr

    return {
        'cs2_wo_timecorr': cs2_wo_timecorr,
        'cs2_wo_picorr': cs2_top_wo_picorr + cs2_bottom_wo_picorr,
        'cs2_wo_elifecorr': cs2_wo_elifecorr,
        'cs2_area_fraction_top_wo_elifecorr': cs2_top_wo_elifecorr / cs2_wo_elifecorr
    }


def _determine_s2_map_names(s2_xy_map):
    """Determine map names for S2 top and bottom corrections."""
    if len(s2_xy_map.map_names) > 1:
        return "map_top", "map_bottom"
    else:
        return "map", "map"


def _ab_region(x, y):
    """Define the AB region based on rotated coordinates."""
    new_x, new_y = straxen.rotate_perp_wires(x, y)
    return (new_x < 28) & (new_x > -28) & (new_x**2 + new_y**2 < 60**2)


def _cd_region(x, y):
    """Define the CD region as the complement of AB."""
    return ~_ab_region(x, y)
