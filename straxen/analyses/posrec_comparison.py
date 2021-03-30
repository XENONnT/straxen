import numpy as np
import pandas as pd

import strax
import straxen

export, __all__ = strax.exporter()


@straxen.mini_analysis(requires=('event_basics', 'event_posrec_many',))
def load_corrected_positions(context, run_id, events,
                             cmt_version=None,
                             posrec_algos=('mlp', 'gcn', 'cnn')):

    """
    Returns the corrected position for each position algorithm available,
        without the need to reprocess event_basics, as the needed
        information is already stored in event_posrec_many.
    
    :param cmt_version: CMT version to use (it can be a list of same
        length as posrec_algos, if different versions are required for
        different posrec algorithms, default 'ONLINE')
    :param posrec_algos: list of position reconstruction algorithms to
        use (default ['mlp', 'gcn', 'cnn'])
    """
    
    posrec_algos = strax.to_str_tuple(posrec_algos)
    
    if cmt_version is None:
        fdc_config = None
        try:
            fdc_config = context.get_single_plugin(run_id, 'event_positions').config['fdc_map']
            cmt_version = fdc_config[1][1]
        except IndexError as e:
            raise ValueError(f'CMT is not set? Your fdc config is {fdc_config}') from e
    
    if hasattr(cmt_version, '__len__') and not isinstance(cmt_version, str) and len(cmt_version) != len(posrec_algos):
        raise TypeError(f"cmt_version is a list but does not match the posrec_algos ({posrec_algos}) length.")
        
    cmt_version = (cmt_version, ) * len(posrec_algos) if isinstance(cmt_version, str) else cmt_version
    
    dtype = []
    
    for algo in posrec_algos:

        dtype += [
                ((f'Interaction x-position, field-distortion corrected (cm) - {algo.upper()} posrec algorithm', f'x_{algo}'), 
                 np.float32),
                ((f'Interaction y-position, field-distortion corrected (cm) - {algo.upper()} posrec algorithm', f'y_{algo}'), 
                 np.float32),
                ((f'Interaction z-position, field-distortion corrected (cm) - {algo.upper()} posrec algorithm', f'z_{algo}'), 
                 np.float32),
                ((f'Interaction radial position, field-distortion corrected (cm) - {algo.upper()} posrec algorithm', f'r_{algo}'),
                 np.float32),
                ((f'Interaction r-position using observed S2 positions directly (cm) - {algo.upper()} posrec algorithm', f'r_naive_{algo}'),
                 np.float32),
                ((f'Correction added to r_naive for field distortion (cm) - {algo.upper()} posrec algorithm',
                  f'r_field_distortion_correction_{algo}'), np.float32),
                ((f'Interaction angular position (radians) - {algo.upper()} posrec algorithm', f'theta_{algo}'),
                 np.float32)]
        
    dtype += [(('Interaction z-position using mean drift velocity only (cm)', 'z_naive'), np.float32)]
    result = np.zeros(len(events), dtype=dtype)

    drift_speed = context.get_single_plugin(run_id, 'event_positions').config['electron_drift_velocity']
    z_obs = - drift_speed * events['drift_time']
    
    for algo, v_cmt in zip(posrec_algos, cmt_version):
        fdc_tmp = ('CMT_model', (f'fdc_map_{algo}', v_cmt), True)
        map_tmp = straxen.get_config_from_cmt(run_id, fdc_tmp)
        itp_tmp = straxen.InterpolatingMap(straxen.common.get_resource(map_tmp, fmt='binary'))
        itp_tmp.scale_coordinates([1., 1., -drift_speed])

        orig_pos = np.vstack([events[f's2_x_{algo}'], events[f's2_y_{algo}'], z_obs]).T
        r_obs = np.linalg.norm(orig_pos[:, :2], axis=1)
        delta_r = itp_tmp(orig_pos)

        # apply radial correction
        with np.errstate(invalid='ignore', divide='ignore'):
            r_cor = r_obs + delta_r
            scale = r_cor / r_obs

        with np.errstate(invalid='ignore'):
            z_cor = -(z_obs ** 2 - delta_r ** 2) ** 0.5
            invalid = np.abs(z_obs) < np.abs(delta_r)
        z_cor[invalid] = z_obs[invalid]

        result[f'x_{algo}'] = orig_pos[:, 0] * scale
        result[f'y_{algo}'] = orig_pos[:, 1] * scale
        result[f'r_{algo}'] = r_cor
        result[f'r_naive_{algo}'] = r_obs
        result[f'r_field_distortion_correction_{algo}'] = delta_r
        result[f'theta_{algo}'] = np.arctan2(orig_pos[:, 1], orig_pos[:, 0])
        result[f'z_{algo}'] = z_cor

    result['z_naive'] = z_obs
    return result
