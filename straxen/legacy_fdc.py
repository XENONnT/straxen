   
import strax

from straxen.common import get_resource, pax_file, first_sr1_run
from straxen.itp_map import InterpolatingMap
from .url_config import URLConfig


export, __all__ = strax.exporter()


RUN_MAPPINGS = {
    "xenon1t_sr0_sr1": [
    (0, pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz')),  # noqa
    (first_sr1_run, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part1_v1.json.gz')), # noqa
    (170411_0611, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part2_v1.json.gz')), # noqa
    (170704_0556, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part3_v1.json.gz')), # noqa
    (170925_0622, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part4_v1.json.gz')), # noqa
    (1_000_000_000_000, None), # noqa
    ]
}

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
