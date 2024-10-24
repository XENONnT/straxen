from typing import Tuple

import numpy as np
import strax
import straxen
from straxen.common import rotate_perp_wires
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO
from straxen.peak_correction_functions import infer_correction_dtype
from straxen.peak_correction_functions import apply_all_corrections
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

    __version__ = "0.5.5"

    depends_on: Tuple[str, ...] = ("event_basics", "event_positions")

    
    # Descriptor configs
    elife = straxen.URLConfig(
        default="cmt://elife?version=ONLINE&run_id=plugin.run_id", help="electron lifetime in [ns]"
    )

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )
    s1_xyz_map = straxen.URLConfig(
        default=(
            "itp_map://resource://cmt://format://"
            "s1_xyz_map_{algo}?version=ONLINE&run_id=plugin.run_id"
            "&fmt=json&algo=plugin.default_reconstruction_algorithm"
        ),
        cache=True,
    )
    s2_xy_map = straxen.URLConfig(
        default=(
            "itp_map://resource://cmt://format://"
            "s2_xy_map_{algo}?version=ONLINE&run_id=plugin.run_id"
            "&fmt=json&algo=plugin.default_reconstruction_algorithm"
        ),
        cache=True,
    )

    # average SE gain for a given time period. default to the value of this run in ONLINE model
    # thus, by default, there will be no time-dependent correction according to se gain
    avg_se_gain = straxen.URLConfig(
        default="cmt://avg_se_gain?version=ONLINE&run_id=plugin.run_id",
        help=(
            "Nominal single electron (SE) gain in PE / electron extracted. "
            "Data will be corrected to this value"
        ),
    )

    # se gain for this run, allowing for using CMT. default to online
    se_gain = straxen.URLConfig(
        default="cmt://se_gain?version=ONLINE&run_id=plugin.run_id",
        help="Actual SE gain for a given run (allows for time dependence)",
    )

    # relative extraction efficiency which can change with time and modeled by CMT.
    rel_extraction_eff = straxen.URLConfig(
        default="cmt://rel_extraction_eff?version=ONLINE&run_id=plugin.run_id",
        help="Relative extraction efficiency for this run (allows for time dependence)",
    )

    # relative light yield
    # defaults to no correction
    rel_light_yield = straxen.URLConfig(
        default="cmt://relative_light_yield?version=ONLINE&run_id=plugin.run_id",
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
        return infer_correction_dtype()

    def compute(self, events):
        result = np.zeros(len(events), self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        # Apply all corrections from peak_corrections.py
        correction_result = apply_all_corrections(self, events)
        for key, value in correction_result.items():
            result[key] = value

        return result
