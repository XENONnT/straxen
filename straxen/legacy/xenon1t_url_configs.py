"""Support for 1T per_run_default options"""

import strax
import straxen
from straxen.common import get_resource, pax_file, first_sr1_run
from straxen.itp_map import InterpolatingMap
from straxen.url_config import URLConfig
import numpy as np

export, __all__ = strax.exporter()

FIXED_TO_PE = {
    'to_pe_placeholder': np.repeat(0.0085, straxen.n_tpc_pmts),
    '1T_to_pe_placeholder': np.array(
        [0.007, 0., 0., 0.008, 0.004, 0.008, 0.004, 0.008, 0.007, 0.005, 0.007, 0.006, 0., 0.006, 0.008, 0.007, 0.006,
         0.009, 0.007, 0.007, 0.007, 0.012, 0.004, 0.008, 0.005, 0.008, 0., 0., 0.007, 0.007, 0.004, 0., 0.004, 0.007,
         0., 0.005, 0.007, 0.007, 0.005, 0.005, 0.008, 0.006, 0.005, 0.007, 0.006, 0.007, 0.008, 0.005, 0.008, 0.008,
         0.005, 0.005, 0.007, 0.008, 0.005, 0.009, 0.004, 0.005, 0.01, 0.008, 0.006, 0.016, 0., 0.005, 0.005, 0., 0.01,
         0.008, 0.004, 0.006, 0.005, 0., 0.008, 0., 0.004, 0.004, 0.006, 0.005, 0.012, 0., 0.005, 0.004, 0.004, 0.008,
         0.007, 0.012, 0., 0., 0., 0.007, 0.007, 0., 0.005, 0.008, 0.006, 0.004, 0.004, 0.006, 0.008, 0.008, 0.008,
         0.006, 0., 0.007, 0.005, 0.005, 0.005, 0.007, 0.004, 0.008, 0.007, 0.008, 0.008, 0.006, 0.006, 0.01, 0.005,
         0.008, 0., 0.012, 0.007, 0.004, 0.008, 0.007, 0.007, 0.008, 0.003, 0.004, 0.007, 0.006, 0., 0.005, 0.004,
         0.005, 0., 0., 0.004, 0., 0.004, 0., 0.004, 0., 0.011, 0.005, 0.006, 0.005, 0.004, 0.004, 0., 0.007, 0., 0.004,
         0., 0.005, 0.006, 0.007, 0.005, 0.008, 0.004, 0.006, 0.008, 0.007, 0., 0.008, 0.008, 0.007, 0.007, 0., 0.008,
         0.004, 0.004, 0.005, 0.004, 0.007, 0.008, 0.004, 0.006, 0.006, 0., 0.007, 0.004, 0.004, 0.005, 0., 0.008,
         0.004, 0.004, 0.004, 0.008, 0.008, 0., 0.006, 0.005, 0.004, 0.005, 0.008, 0.008, 0.008, 0., 0.005, 0.008, 0.,
         0.008, 0., 0.004, 0.012, 0., 0.005, 0.007, 0.009, 0.005, 0.004, 0.004, 0., 0., 0.004, 0.004, 0.011, 0.004,
         0.004, 0.007, 0.004, 0.005, 0.004, 0.005, 0.007, 0.004, 0.006, 0.006, 0.004, 0.008, 0.005, 0.007, 0.007, 0.,
         0.004, 0.007, 0.008, 0.004, 0., 0.007, 0.004, 0.004, 0.004, 0., 0.004, 0.005, 0.004]),
    # Gains which will preserve all areas in adc counts.
    # Useful for debugging and tests.
    'adc_tpc': np.ones(straxen.n_tpc_pmts),
    'adc_mv': np.ones(straxen.n_mveto_pmts),
    'adc_nv': np.ones(straxen.n_nveto_pmts)
}

RUN_MAPPINGS = {
    "xenon1t_sr0_sr1": [
        (0, pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz')),  # noqa
        (first_sr1_run, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part1_v1.json.gz')),
        # noqa
        (170411_0611, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part2_v1.json.gz')),
        # noqa
        (170704_0556, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part3_v1.json.gz')),
        # noqa
        (170925_0622, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part4_v1.json.gz')),
        # noqa
        (1_000_000_000_000, None),  # noqa
    ]
}


@URLConfig.register('legacy-to-pe')
def get_fixed_pe(name: str):
    """Return a fixed value for a given name"""
    return FIXED_TO_PE[name]


@URLConfig.register('legacy-thresholds')
def get_thresholds(model: str):
    """Return a fixed value for a given model"""

    return straxen.legacy.hit_min_amplitude(model)


@URLConfig.register('legacy-fdc')
def get_legacy_fdc(name, run_id=None):
    if run_id is None:
        raise ValueError('Must provide run_id to get legacy fdc')

    if isinstance(run_id, str):
        run_id = int(run_id.replace('_', ''))

    if name not in RUN_MAPPINGS:
        raise ValueError(f'Unknown legacy fdc name {name}')

    mapping = RUN_MAPPINGS[name]

    for (start_run, url), (end_run, _) in zip(mapping[:-1], mapping[1:]):
        if run_id >= start_run and run_id < end_run:
            break
    else:
        raise ValueError(f'No legacy fdc for run {run_id}')

    if url is None:
        raise ValueError(f'No legacy fdc for run {run_id}')

    return InterpolatingMap(get_resource(url, fmt='binary'))


@URLConfig.register('legacy-z_bias')
def get_z_bias(offset: str):
    """Return a lambda function return offset as placeholder"""
    def fake_z_bias(rz, **kwargs):
        return np.zeros(len(rz)) * int(offset)

    return fake_z_bias
