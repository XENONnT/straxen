import numpy as np

import strax
import straxen


export, __all__ = strax.exporter()


@export
def hit_min_amplitude(model):
    """Return hitfiner height threshold to use in processing
    
    :param model: Model name (str), or int to use a uniform threshold,
    or array/tuple or thresholds to use.
    """
    
    if isinstance(model, (int, float)):
        return np.ones(straxen.n_tpc_pmts, dtype=np.int16) * model

    if isinstance(model, (tuple, np.ndarray)):
        return model
    
    if model == 'XENON1T_SR1':
        return np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 18, 15, 15, 15, 15, 15, 54, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 15, 15, 35, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 18, 15, 15, 15, 15, 15, 15, 15, 17, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 17, 15, 15, 26, 88, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 20, 22, 15, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 17, 15, 15, 15, 15, 15, 17, 16, 15, 15, 15, 15, 15, 15, 17, 16, 15, 15, 15, 15, 15, 15, 45, 15, 15, 15, 15, 25, 15, 15, 15, 17, 15, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 24, 15, 17, 15, 15, 18, 15, 15, 15, 34, 15, 15, 18, 15, 15, 39, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 18, 15, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 15, 15, 19, 15, 15, 15, 15, 15, 15, 17, 15, 15, 18, 15, 15, 15, 15, 15, 17, 15, 18, 15, 15, 15, 17, 15, 18, 15, 35, 15, 15], dtype=np.int16)
    
    if model == 'pmt_commissioning_initial':
        # ADC thresholds used for the initial PMT commissioning data
        # (at least since April 28 2020, run 007305)
        result = 15 * np.ones(straxen.n_tpc_pmts, dtype=np.int16)
        result[453] = 30
        return result
    
    if model == 'pmt_commissioning_initial_he':
        # ADC thresholds used for the initial PMT commissioning data
        # (at least since April 28 2020, run 007305)
        result = 15 * np.ones(straxen.contexts.xnt_common_config['channel_map']['he'][1], dtype=np.int16)
        return result
    
    raise ValueError(f"Unknown ADC threshold model {model}")
