from typing import Tuple

import numpy as np
import strax
import straxen
from straxen.common import rotate_perp_wires
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO

export, __all__ = strax.exporter()


@export
class CorrectedAreas(strax.Plugin):
    """Plugin which applies light collection efficiency maps and electron life time to the data.

    Computes the cS1/cS2 for the main/alternative S1/S2 as well as the
    corrected life time.
    Note:
        Please be aware that for both, the main and alternative S1, the
        area is corrected according to the xy-position of the main S2.
        There are now 3 components of cS2s: cs2_top, cS2_bottom and cs2.
        cs2_top and cs2_bottom are corrected by the corresponding maps,
        and cs2 is the sum of the two.

    N-1 corrections are also provided, where each variable has all corrections
    applied except for one specific correction. This allows studying the impact
    of individual corrections.

    The following corrections are applied:
    - Peak reconstruction bias correction (corrects for bias in peak finding algorithm)
    - S1 xyz correction (light collection efficiency)
    - S2 xy correction (light collection efficiency)
    - Time-dependent light yield correction
    - Single electron gain (SEG) and extraction efficiency (EE) correction (partition,time)
    - Photoionization correction for S2 bottom
    - Electron lifetime correction
    - Time-dependent charge yield correction

    """

    __version__ = "0.5.9"

    depends_on: Tuple[str, ...] = ("event_basics", "event_positions")

    # Descriptor configs
    elife = straxen.URLConfig(
        default="xedocs://electron_lifetimes?attr=value&run_id=plugin.run_id&version=ONLINE",
        help="electron lifetime in [ns]",
    )

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )
    s1_xyz_map = straxen.URLConfig(
        default="xedocs://s1_xyz_maps"
        "?run_id=plugin.run_id"
        "&algorithm=plugin.default_reconstruction_algorithm&attr=map&version=ONLINE",
        cache=True,
    )
    s2_xy_map = straxen.URLConfig(
        default="xedocs://s2_xy_maps"
        "?run_id=plugin.run_id"
        "&algorithm=plugin.default_reconstruction_algorithm&attr=map&version=ONLINE",
        cache=True,
    )

    # average SE gain for a given time period. default to the value of this run in ONLINE model
    # thus, by default, there will be no time-dependent correction according to se gain
    avg_se_gain = straxen.URLConfig(
        default="xedocs://avg_se_gains?run_id=plugin.run_id&version=ONLINE&attr=value",
        help="Nominal single electron (SE) gain in PE / electron extracted. "
        "Data will be corrected to this value",
    )

    # se gain for this run, allowing for using xedocs. default to online
    se_gain = straxen.URLConfig(
        default="take://objects-to-dict://"
        "xedocs://se_gains"
        "?partition=all_tpc&run_id=plugin.run_id&sort=partition"
        "&as_list=True&key_attr=partition&value_attr=value&take=all_tpc&version=ONLINE",
        help="Actual SE gain for a given run (allows for time dependence)",
    )

    # relative extraction efficiency which can change with time and modeled by xedocs.
    rel_extraction_eff = straxen.URLConfig(
        default="take://objects-to-dict://"
        "xedocs://rel_extraction_effs"
        "?partition=all_tpc&run_id=plugin.run_id&sort=partition"
        "&as_list=True&key_attr=partition&value_attr=value&take=all_tpc&version=ONLINE",
        help="Relative extraction efficiency for this run (allows for time dependence)",
    )

    # relative charge yield
    # defaults to no correction
    rel_charge_yield = straxen.URLConfig(default=1, help="Relative charge yield for SR2 only")

    # relative light yield
    # defaults to no correction
    rel_light_yield = straxen.URLConfig(
        default="xedocs://relative_light_yield?attr=value&run_id=plugin.run_id&version=ONLINE",
        help="Relative light yield (allows for time dependence)",
    )

    # b parameter for z-dependent relative light yield correction
    b_rel_light_yield = straxen.URLConfig(
        default=0.0,
        help="b parameter for z-dependent relative light yield correction",
    )

    # slope parameter for z-dependent relative light yield correction
    slope_rel_light_yield = straxen.URLConfig(
        default=0.0,
        help="Slope parameter for z-dependent relative light yield correction",
    )

    # Single electron gain partition
    # AB and CD partitons distiguished based on
    # linear and circular regions
    # SR0 values set as default
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:sr0_2_region_se_correction
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:noahhood:corrections:se_gain_ee_final
    single_electron_gain_partition = straxen.URLConfig(
        default={"linear": 28, "circular": 60},
        help=(
            "Two distinct patterns of evolution of single electron corrections between AB and CD. "
            "Distinguish thanks to linear and circular regions"
        ),
    )

    # cS2 AFT correction due to photoionization
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:zihao:sr1_s2aft_photonionization_correction
    cs2_bottom_top_ratio_correction = straxen.URLConfig(
        default=1, help="Scaling factor for cS2 AFT correction due to photoionization"
    )

    # S1 Peak Reconstruction Bias Map
    s1_bias_map = straxen.URLConfig(
        default="itp_map://resource://xedocs://peak_reconstruction_bias"
        "?attr=value&run_id=plugin.run_id&signal=s1&fmt=json&version=ONLINE",
        help="Interpolation map for S1 peak bias correction. "
        "Bias is defined as (reconstructed / raw) - 1."
        "So, the bias correction is reconstructed / (1 + bias).",
    )

    # S2 Peak Reconstruction Bias Map
    s2_bias_map = straxen.URLConfig(
        default="itp_map://resource://xedocs://peak_reconstruction_bias"
        "?attr=value&run_id=plugin.run_id&signal=s2&fmt=json&version=ONLINE",
        help="Interpolation map for S2 peak bias correction. "
        "Bias is defined as (reconstructed / raw) - 1."
        "So, the bias correction is reconstructed / (1 + bias).",
    )

    check_s2_only_aft = straxen.URLConfig(
        default=True, type=bool, track=False, help="Whether to check NaN AFT of S2-Only events"
    )

    # Intermediate S2 corrections for studying correction order
    # Correction factors, listed in the order of application:
    # 1. Peak bias correction
    # 2. S2 xy position correction
    # 3. SEG/EE correction
    # 4. Photoionization correction for S2 bottom
    # 5. Electron lifetime correction
    # 6. Relative charge yield correction
    # Encode included corrections in binary strings
    # (because that is easier to read than big lists of bools).
    # E.g. '010010' means correcting for S2xy and elife, but not for
    # peak bias, SEG/EE, PI and relCY
    # '111111' is the fully corrected cS2.

    name_postfixes = ["_bias", "_xy", "_segee", "_pi", "_relcy", "_elife"]
    description_strs = [
        "peak bias",
        "S2 xy",
        "SEG/EE",
        "photoionization",
        "relative charge yield",
        "elife",
    ]
    intermediate_cs2s = [
        "111111",
        "100000",
        "110000",
        "101000",
        "111000",
        "110001",
        "000001",
        "010001",
        "011101",
        "101101",
        "110101",
        "111001",
        "111100",
        "111101",
        "111110",
    ]

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for peak_type, peak_name in zip(["", "alt_"], ["main", "alternate"]):
            # S1 corrections
            dtype += [
                (f"{peak_type}cs1", np.float32, f"Corrected area of {peak_name} S1 [PE]"),
                (
                    f"{peak_type}cs1_wo_timecorr",
                    np.float32,
                    f"Corrected area of {peak_name} S1 (without rel LY evolution correction) [PE]",
                ),
                (
                    f"{peak_type}cs1_wo_xyzcorr",
                    np.float32,
                    f"Corrected area of {peak_name} S1 (without xyz position correction) [PE]",
                ),
                (
                    f"{peak_type}cs1_wo_peakbiascorr",
                    np.float32,
                    f"Corrected area of {peak_name} S1 (without peak bias correction) [PE]",
                ),
            ]

            for encoding in self.intermediate_cs2s:
                # if all corrections are included its the final cS2
                if encoding == "111111":
                    postfix = ""
                    description = ""
                else:
                    postfix = "_w"
                    including = []
                    excluding = []
                    for i_c, char in enumerate(encoding):
                        if int(char):
                            postfix += self.name_postfixes[i_c]
                            including.append(self.description_strs[i_c])
                        else:
                            excluding.append(self.description_strs[i_c])
                    inc_str = " + ".join(including)
                    exc_str = " + ".join(excluding)
                    description = f" (including {inc_str}, excluding {exc_str})"

                main_comment = f"Corrected area of {peak_name} S2{description} [PE]"
                aft_comment = (
                    f"Fraction of area seen by the top PMT array for corrected "
                    f"{peak_name} S2{description}"
                )

                dtype += [
                    (f"{peak_type}cs2{postfix}", np.float32, main_comment),
                    (f"{peak_type}cs2_area_fraction_top{postfix}", np.float32, aft_comment),
                ]
        return dtype

    def ab_region(self, x, y):
        new_x, new_y = rotate_perp_wires(x, y)
        cond = new_x < self.single_electron_gain_partition["linear"]
        cond &= new_x > -self.single_electron_gain_partition["linear"]
        cond &= new_x**2 + new_y**2 < self.single_electron_gain_partition["circular"] ** 2
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
        """Get single electron gain and extraction efficiency options."""
        self.regions = {"ab": self.ab_region, "cd": self.cd_region}

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

    def rel_light_yield_correction(self, events):
        """Compute relative light yield correction (z- and t-dependent)."""

        a = self.slope_rel_light_yield * (self.rel_light_yield - 1)
        b = self.b_rel_light_yield

        # Compute full z- and t-dependent correction
        rel_ly_zt_corr = self.rel_light_yield * (a * (events["z"] ** 2 + b * events["z"]) + 1)

        return rel_ly_zt_corr

    def apply_s2_corrections(
        self,
        s2_area,
        s2_aft,
        s2_bias_correction,
        s2_xy_correction_top,
        s2_xy_correction_bottom,
        seg_ee_corr,
        pi_corr_bottom,
        elife_correction,
        rel_cy_correction,
    ):
        """Apply S2 corrections and return various corrected areas. To study the impact of
        individual corrections, parameters of this function can be set to 1, thereby excluding the
        corresponding correction.

        Returns:
            cs2,
            cs2_area_fraction_top

        """
        # Base areas
        s2_area_top = s2_area * s2_aft
        s2_area_bottom = s2_area * (1 - s2_aft)

        # Apply peak bias, S2 xy and SEG/EE to top and bottom
        cs2_top_wo_elife = s2_area_top / s2_bias_correction / s2_xy_correction_top / seg_ee_corr
        cs2_bottom_wo_elife = (
            s2_area_bottom / s2_bias_correction / s2_xy_correction_bottom / seg_ee_corr
        )

        # Apply rel_cy_correction
        cs2_relcy = (cs2_top_wo_elife + cs2_bottom_wo_elife) * rel_cy_correction

        # Apply elife to get total cS2
        cs2 = cs2_relcy * elife_correction

        # Apply PI AFT correction to get cAFT
        # Do this on the cS2 without elife, because S2-only events have NaN as elife,
        # and elife cancels out in the AFT fraction anyway.
        cs2_area_fraction_top = cs2_top_wo_elife / (
            cs2_top_wo_elife + cs2_bottom_wo_elife * pi_corr_bottom
        )

        return (
            cs2,
            cs2_area_fraction_top,
        )

    def compute(self, events):
        result = np.zeros(len(events), self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        # S1 corrections depend on the actual corrected event position.
        # We use this also for the alternate S1; for e.g. Kr this is
        # fine as the S1 correction varies slowly.
        event_positions = np.vstack([events["x"], events["y"], events["z"]]).T

        # S1 corrections
        for peak_type in ["", "alt_"]:
            # Correction factors
            s1_area = events[f"{peak_type}s1_area"]
            s1_bias_correction = 1 + self.s1_bias_map(s1_area.reshape(-1, 1)).flatten()
            s1_xyz_correction = self.s1_xyz_map(event_positions)
            s1_time_correction = self.rel_light_yield_correction(events)

            # Apply all corrections
            result[f"{peak_type}cs1"] = (
                s1_area / s1_bias_correction / s1_xyz_correction / s1_time_correction
            )

            # N-1 corrections for S1
            result[f"{peak_type}cs1_wo_peakbiascorr"] = (
                s1_area / s1_xyz_correction / s1_time_correction
            )
            result[f"{peak_type}cs1_wo_xyzcorr"] = s1_area / s1_bias_correction / s1_time_correction
            result[f"{peak_type}cs1_wo_timecorr"] = s1_area / s1_bias_correction / s1_xyz_correction

        # S2 corrections
        s2_top_map_name, s2_bottom_map_name = self.s2_map_names()
        seg, avg_seg, ee = self.seg_ee_correction_preparation()

        for peak_type in ["", "alt_"]:
            s2_area = events[f"{peak_type}s2_area"]
            s2_aft = events[f"{peak_type}s2_area_fraction_top"]
            s2_positions = np.vstack([events[f"{peak_type}s2_x"], events[f"{peak_type}s2_y"]]).T

            # Correction factors, listed in the order of application:
            # 1. Peak bias correction
            # 2. S2 xy position correction
            # 3. SEG/EE correction
            # 4. Photoionization correction for S2 bottom
            # 5. Relative charge yield correction
            # 6. Electron lifetime correction
            # Must make sure that the elife is the last
            # because it will cause AFT nan for S2Only events

            s2_bias_correction = 1 + self.s2_bias_map(s2_area.reshape(-1, 1)).flatten()
            s2_xy_correction_top = self.s2_xy_map(s2_positions, map_name=s2_top_map_name)
            s2_xy_correction_bottom = self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name)
            rel_cy_correction_factor = self.rel_charge_yield

            seg_ee_corr = np.zeros(len(events))
            for partition, func in self.regions.items():
                mask = func(events[f"{peak_type}s2_x"], events[f"{peak_type}s2_y"])
                seg_ee_corr[mask] = seg[partition] / avg_seg[partition] * ee[partition]

            pi_corr_bottom = self.cs2_bottom_top_ratio_correction

            el_string = peak_type + "s2_interaction_" if peak_type == "alt_" else peak_type
            elife_correction = np.exp(events[f"{el_string}drift_time"] / self.elife)

            corrections_parameters = [
                s2_bias_correction,
                s2_xy_correction_top,
                s2_xy_correction_bottom,
                seg_ee_corr,
                pi_corr_bottom,
                rel_cy_correction_factor,
                elife_correction,
            ]

            for encoding in self.intermediate_cs2s:
                postfix = "_w"
                # Set correction parameters that are not included in the encoding to 1
                # Note that S2xy has 2 parameters, therefore this list has len 7
                _correction_parameters = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                for i_c, char in enumerate(encoding):
                    if int(char):
                        postfix += self.name_postfixes[i_c]
                        # S2xy has 2 values...
                        if i_c == 0:
                            _correction_parameters[i_c] = corrections_parameters[i_c]
                        elif i_c == 1:
                            _correction_parameters[i_c] = corrections_parameters[i_c]
                            _correction_parameters[i_c + 1] = corrections_parameters[i_c + 1]
                        else:
                            _correction_parameters[i_c + 1] = corrections_parameters[i_c + 1]
                # No postfix for fully corrected
                if encoding == "111111":
                    postfix = ""
                (
                    result[f"{peak_type}cs2{postfix}"],
                    result[f"{peak_type}cs2_area_fraction_top{postfix}"],
                ) = self.apply_s2_corrections(
                    s2_area,
                    s2_aft,
                    *_correction_parameters,
                )

        if self.check_s2_only_aft:
            s2_only = np.isnan(events["s1_area"])
            s2_only &= ~np.isnan(result["cs2"])
            if np.any(np.isnan(result["cs2_area_fraction_top"][s2_only])):
                raise ValueError(
                    "NaN AFT for S2-Only events! "
                    "Even for S2-Only events (w/o cS2), the AFT should be defined."
                )

        return result
