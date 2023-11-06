"""Some shared defaults"""
DEFAULT_POSREC_ALGO = 'mlp'

HE_PREAMBLE = """High energy channels: attenuated signals of the top PMT-array\n"""

MV_PREAMBLE = 'Muno-Veto Plugin: Same as the corresponding nVETO-PLugin.\n'

NV_HIT_DEFAULTS = {
    'save_outside_hits_nv': (3, 15),
    'hit_min_amplitude_nv': 'xedocs://hit_thresholds?version=ONLINE&run_id=plugin.run_id&detector=neutron_veto',
}

MV_HIT_DEFAULTS = {
    'save_outside_hits_mv': (2, 5),
    'hit_min_amplitude_mv': 'xedocs://hit_thresholds_mv?version=ONLINE&run_id=plugin.run_id&detector=muon_veto',
}

FAKE_MERGED_S2_TYPE = -42
