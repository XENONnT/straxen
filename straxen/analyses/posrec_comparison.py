import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@straxen.mini_analysis(requires=('event_basics',))
def load_corrected_positions(context, run_id, events,
                             cmt_version=None,
                             posrec_algos=('mlp', 'gcn', 'cnn')):
    """
    Returns the corrected position for each position algorithm available,
        without the need to reprocess event_basics, as the needed
        information is already stored in event_basics.
    
    :param cmt_version: CMT version to use (it can be a list of same
        length as posrec_algos, if different versions are required for
        different posrec algorithms, default 'local_ONLINE')
    :param posrec_algos: list of position reconstruction algorithms to
        use (default ['mlp', 'gcn', 'cnn'])
    """

    posrec_algos = strax.to_str_tuple(posrec_algos)

    if cmt_version is None:
        fdc_config = context.get_single_plugin(run_id, 'event_positions').config['fdc_map']
        if isinstance(fdc_config, str) and 'cmt://' in fdc_config:
            cmt_version = straxen.URLConfig.split_url_kwargs(fdc_config)
        elif straxen.is_cmt_option(fdc_config):
            cmt_version = fdc_config[1]
        else:
            raise ValueError('FDC map is not a CMT option, cannot infer cmt version.')

    if (
            isinstance(cmt_version, (tuple, list))
            and len(cmt_version) != len(posrec_algos)
    ):
        raise TypeError(f"cmt_version is a list but does not match the "
                        f"posrec_algos ({posrec_algos}) length.")

    cmt_version = ((cmt_version, ) * len(posrec_algos)
                   if isinstance(cmt_version, str) else cmt_version)

    # Get drift from CMT
    ep = context.get_single_plugin(run_id, 'event_positions')
    drift_speed = ep.electron_drift_velocity
    dtype = []

    for algo in posrec_algos:
        for xyzr in 'x y z r'.split():
            dtype += [
                ((f'Interaction {xyzr}-position, field-distortion corrected (cm) - '
                  f'{algo.upper()} posrec algorithm', f'{xyzr}_{algo}'),
                 np.float32),
            ]
        dtype += [
            ((f'Interaction r-position using observed S2 positions directly (cm) -'
              f' {algo.upper()} posrec algorithm', f'r_naive_{algo}'),
             np.float32),
            ((f'Correction added to r_naive for field distortion (cm) - '
              f'{algo.upper()} posrec algorithm',
              f'r_field_distortion_correction_{algo}'), np.float32),
            ((f'Interaction angular position (radians) - {algo.upper()} '
              f'posrec algorithm', f'theta_{algo}'),
             np.float32)]

    dtype += [(('Interaction z-position using mean drift velocity only (cm)', 'z_naive'), np.float32)]
    result = np.zeros(len(events), dtype=dtype)

    z_obs = - drift_speed * events['drift_time']

    for algo, v_cmt in zip(posrec_algos, cmt_version):
        fdc_tmp = (f'fdc_map_{algo}', v_cmt, True)
        map_tmp = straxen.get_correction_from_cmt(run_id, fdc_tmp)
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
