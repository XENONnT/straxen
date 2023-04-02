import numpy as np
import straxen
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO

import strax

export, __all__ = strax.exporter()


@export
class EventPositions(strax.Plugin):
    """
    Computes the observed and corrected position for the main S1/S2
    pairs in an event. For XENONnT data, it returns the FDC corrected
    positions of the default_reconstruction_algorithm. In case the fdc_map
    is given as a file (not through CMT), then the coordinate system
    should be given as (x, y, z), not (x, y, drift_time).
    """

    depends_on = ('event_basics',)

    __version__ = '0.2.0'

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO,
        help="default reconstruction algorithm that provides (x,y)"
    )

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    electron_drift_time_gate = straxen.URLConfig(
        default='cmt://'
                'electron_drift_time_gate'
                '?version=ONLINE&run_id=plugin.run_id',
        help='Electron drift time from the gate in ns',
        cache=True)

    fdc_map = straxen.URLConfig(
        infer_type=False,
        help='3D field distortion correction map path',
        default='legacy-fdc://xenon1t_sr0_sr1?run_id=plugin.run_id')

    z_bias_map = straxen.URLConfig(
        infer_type=False,
        help='Map of Z bias due to non uniform drift velocity/field',
        default='legacy-z_bias')

    def infer_dtype(self):
        dtype = []
        for j in 'x y r'.split():
            comment = f'Main interaction {j}-position, field-distortion corrected (cm)'
            dtype += [(j, np.float32, comment)]
            for s_i in [1, 2]:
                comment = f'Alternative S{s_i} interaction (rel. main S{int(2 * (1.5 - s_i) + s_i)}) {j}-position, field-distortion corrected (cm)'
                field = f'alt_s{s_i}_{j}_fdc'
                dtype += [(field, np.float32, comment)]

        for j in ['z']:
            comment = 'Interaction z-position, using mean drift velocity only (cm)'
            dtype += [(j, np.float32, comment)]
            comment = 'Interaction z-position corrected to non-uniform drift velocity [ cm ]'
            dtype += [(j + "_dv_corr", np.float32, comment)]
            for s_i in [1, 2]:
                comment = f'Alternative S{s_i} z-position (rel. main S{int(2 * (1.5 - s_i) + s_i)}), using mean drift velocity only (cm)'
                field = f'alt_s{s_i}_z'
                dtype += [(field, np.float32, comment)]
                # values for corrected Z position
                comment = f'Alternative S{s_i} z-position (rel. main S{[1 if s_i==2 else 2]}), corrected for non-uniform field (cm)'
                field = f'alt_s{s_i}_z_dv_corr'
                dtype += [(field, np.float32, comment)]


        naive_pos = []
        fdc_pos = []
        for j in 'r z'.split():
            naive_pos += [(f'{j}_naive',
                           np.float32,
                           f'Main interaction {j}-position with observed position (cm)')]
            fdc_pos += [(f'{j}_field_distortion_correction',
                         np.float32,
                         f'Correction added to {j}_naive for field distortion (cm)')]
            for s_i in [1, 2]:
                naive_pos += [(
                    f'alt_s{s_i}_{j}_naive',
                    np.float32,
                    f'Alternative S{s_i} interaction (rel. main S{int(2 * (1.5 - s_i) + s_i)}) {j}-position with observed position (cm)')]
                fdc_pos += [(f'alt_s{s_i}_{j}_field_distortion_correction',
                             np.float32,
                             f'Correction added to alt_s{s_i}_{j}_naive for field distortion (cm)')]
        dtype += naive_pos + fdc_pos
        for s_i in [1, 2]:
            dtype += [(f'alt_s{s_i}_theta',
                       np.float32,
                       f'Alternative S{s_i} (rel. main S{int(2 * (1.5 - s_i) + s_i)}) interaction angular position (radians)')]

        dtype += [('theta', np.float32, f'Main interaction angular position (radians)')]
        return dtype + strax.time_fields

    def setup(self):
        self.coordinate_scales = [1., 1., - self.electron_drift_velocity]
        self.map = self.fdc_map

    def compute(self, events):

        result = {'time': events['time'],
                  'endtime': strax.endtime(events)}

        # s_i == 0 indicates the main event, while s_i != 0 means alternative S1 or S2 is used based on s_i value
        # while the other peak is the main one (e.g., s_i == 1 means that the event is defined using altS1 and main S2)
        for s_i in [0, 1, 2]:
            # alt_sx_interaction_drift_time is calculated between main Sy and alternative Sx
            drift_time = events['drift_time'] if not s_i else events[f'alt_s{s_i}_interaction_drift_time']

            z_obs = - self.electron_drift_velocity * drift_time
            xy_pos = 's2_' if s_i != 2 else 'alt_s2_'
            orig_pos = np.vstack([events[f'{xy_pos}x'], events[f'{xy_pos}y'], z_obs]).T
            r_obs = np.linalg.norm(orig_pos[:, :2], axis=1)
            delta_r = self.map(orig_pos)
            z_obs = z_obs + self.electron_drift_velocity * self.electron_drift_time_gate

            # apply radial correction
            with np.errstate(invalid='ignore', divide='ignore'):
                r_cor = r_obs + delta_r
                scale = np.divide(r_cor, r_obs, out=np.zeros_like(r_cor), where=r_obs != 0)

            # z correction due to longer drift time for distortion
            # calculated based on the Pythagorean theorem where
            # the electron track is assumed to be a straight line
            # (geometrical reasoning not valid if |delta_r| > |z_obs|,
            #  as cathetus cannot be longer than hypothenuse)
            with np.errstate(invalid='ignore'):
                z_cor = -(z_obs ** 2 - delta_r ** 2) ** 0.5
                invalid = np.abs(z_obs) < np.abs(delta_r)
                # do not apply z correction above gate
                invalid |= z_obs >= 0
            z_cor[invalid] = z_obs[invalid]
            delta_z = z_cor - z_obs
            # correction of z bias due to non-uniform field
            z_dv_delta = self.z_bias_map(np.array([r_obs, z_obs]).T, map_name='z_bias_map')

            pre_field = '' if s_i == 0 else f'alt_s{s_i}_'
            post_field = '' if s_i == 0 else '_fdc'
            result.update({f'{pre_field}x{post_field}': orig_pos[:, 0] * scale,
                           f'{pre_field}y{post_field}': orig_pos[:, 1] * scale,
                           f'{pre_field}r{post_field}': r_cor,
                           f'{pre_field}r_naive': r_obs,
                           f'{pre_field}r_field_distortion_correction': delta_r,
                           f'{pre_field}theta': np.arctan2(orig_pos[:, 1], orig_pos[:, 0]),
                           f'{pre_field}z_naive': z_obs,
                           # using z_obs in agreement with the dtype description
                           # the FDC for z (z_cor) is found to be not reliable (see #527)
                           f'{pre_field}z': z_obs,
                           f'{pre_field}z_field_distortion_correction': delta_z,
                           f'{pre_field}z_dv_corr': z_obs - z_dv_delta,
                           })
        return result
