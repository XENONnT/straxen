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

    """

    __version__ = "0.5.3"

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

    # relative light yield
    # defaults to no correction
    rel_light_yield = straxen.URLConfig(
        default="xedocs://relative_light_yield?attr=value&run_id=plugin.run_id&version=ONLINE",
        help="Relative light yield (allows for time dependence)",
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

    # cS2 AFT correction due to photon ionization
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:zihao:sr1_s2aft_photonionization_correction
    cs2_bottom_top_ratio_correction = straxen.URLConfig(
        default=1, help="Scaling factor for cS2 AFT correction due to photon ionization"
    )

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for peak_type, peak_name in zip(["", "alt_"], ["main", "alternate"]):
            # Only apply
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

    def compute(self, events):
        result = np.zeros(len(events), self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        # S1 corrections depend on the actual corrected event position.
        # We use this also for the alternate S1; for e.g. Kr this is
        # fine as the S1 correction varies slowly.
        event_positions = np.vstack([events["x"], events["y"], events["z"]]).T

        for peak_type in ["", "alt_"]:
            result[f"{peak_type}cs1_wo_timecorr"] = events[f"{peak_type}s1_area"] / self.s1_xyz_map(
                event_positions
            )
            result[f"{peak_type}cs1"] = result[f"{peak_type}cs1_wo_timecorr"] / self.rel_light_yield

        # S2 corrections
        s2_top_map_name, s2_bottom_map_name = self.s2_map_names()
        seg, avg_seg, ee = self.seg_ee_correction_preparation()

        # now can start doing corrections
        for peak_type in ["", "alt_"]:
            # S2(x,y) corrections use the observed S2 positions
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

            # collect electron lifetime correction
            # for electron lifetime corrections to the S2s,
            # use drift time computed using the main S1.
            el_string = peak_type + "s2_interaction_" if peak_type == "alt_" else peak_type
            elife_correction = np.exp(events[f"{el_string}drift_time"] / self.elife)

            # collect SEG and EE corrections
            seg_ee_corr = np.zeros(len(events))
            for partition, func in self.regions.items():
                # partitioned SEG and EE
                partition_mask = func(events[f"{peak_type}s2_x"], events[f"{peak_type}s2_y"])
                # correct for SEG and EE
                seg_ee_corr[partition_mask] = seg[partition] / avg_seg[partition] * ee[partition]

            # apply S2 xy correction
            cs2_xycorr = cs2_top_xycorr + cs2_bottom_xycorr
            result[f"{peak_type}cs2_wo_timecorr"] = cs2_xycorr * elife_correction
            result[f"{peak_type}cs2_area_fraction_top_wo_timecorr"] = cs2_top_xycorr / cs2_xycorr

            # apply SEG and EE correction
            cs2_top_wo_picorr = cs2_top_xycorr / seg_ee_corr
            cs2_bottom_wo_picorr = cs2_bottom_xycorr / seg_ee_corr
            cs2_wo_picorr = cs2_top_wo_picorr + cs2_bottom_wo_picorr
            result[f"{peak_type}cs2_wo_picorr"] = cs2_wo_picorr
            result[f"{peak_type}cs2_area_fraction_top_wo_picorr"] = (
                cs2_top_wo_picorr / result[f"{peak_type}cs2_wo_picorr"]
            )

            # apply photon ionization intensity and cS2 AFT correction (see #1247)
            # cS2 bottom should be corrected by photon ionization, but not cS2 top
            cs2_top_wo_elifecorr = cs2_top_wo_picorr
            cs2_bottom_wo_elifecorr = cs2_bottom_wo_picorr * self.cs2_bottom_top_ratio_correction
            cs2_wo_elifecorr = cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr
            # scale top and bottom to ensure total cS2 is conserved, since the time
            # dependence of it has been already corrected by SEG correction
            cs2_top_wo_elifecorr *= cs2_wo_picorr / cs2_wo_elifecorr
            cs2_bottom_wo_elifecorr *= cs2_wo_picorr / cs2_wo_elifecorr
            cs2_wo_elifecorr = cs2_wo_picorr
            result[f"{peak_type}cs2_wo_elifecorr"] = cs2_wo_elifecorr
            result[f"{peak_type}cs2_area_fraction_top_wo_elifecorr"] = (
                cs2_top_wo_elifecorr / result[f"{peak_type}cs2_wo_elifecorr"]
            )

            # apply electron lifetime correction
            result[f"{peak_type}cs2"] = result[f"{peak_type}cs2_wo_elifecorr"] * elife_correction
            result[f"{peak_type}cs2_area_fraction_top"] = result[
                f"{peak_type}cs2_area_fraction_top_wo_elifecorr"
            ]
        return result
