"""Some shared defaults."""

DEFAULT_POSREC_ALGO = "cnf"

HE_PREAMBLE = """High energy channels: attenuated signals of the top PMT-array\n"""

MV_PREAMBLE = "Muno-Veto Plugin: Same as the corresponding nVETO-PLugin.\n"

NV_HIT_DEFAULTS = {
    "save_outside_hits_nv": (3, 15),
    "hit_min_amplitude_nv": (
        "list-to-array://"
        "pad-array://"
        "xedocs://hit_thresholds"
        "?pad_left=2000"
        "&as_list=True"
        "&sort=pmt"
        "&attr=value"
        "&detector=neutron_veto"
        "&run_id=plugin.run_id"
        "&version=ONLINE"
    ),
}

MV_HIT_DEFAULTS = {
    "save_outside_hits_mv": (2, 5),
    "hit_min_amplitude_mv": (
        "list-to-array://"
        "pad-array://"
        "xedocs://hit_thresholds"
        "?pad_left=1000"
        "&as_list=True"
        "&sort=pmt"
        "&attr=value"
        "&detector=muon_veto"
        "&run_id=plugin.run_id"
        "&version=ONLINE"
    ),
}

FAR_XYPOS_S2_TYPE = 20
WIDE_XYPOS_S2_TYPE = 22
