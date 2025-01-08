"""Some shared defaults."""

DEFAULT_POSREC_ALGO = "cnf"

HE_PREAMBLE = """High energy channels: attenuated signals of the top PMT-array\n"""

MV_PREAMBLE = "Muno-Veto Plugin: Same as the corresponding nVETO-PLugin.\n"

NV_HIT_DEFAULTS = {
    "save_outside_hits_nv": (3, 15),
    "hit_min_amplitude_nv": "cmt://hit_thresholds_nv?version=ONLINE&run_id=plugin.run_id",
}

MV_HIT_DEFAULTS = {
    "save_outside_hits_mv": (2, 5),
    "hit_min_amplitude_mv": "cmt://hit_thresholds_mv?version=ONLINE&run_id=plugin.run_id",
}

FAKE_MERGED_S2_TYPE = -42
WIDE_XYPOS_S2_TYPE = 20
