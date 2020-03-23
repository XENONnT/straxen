import strax

import numpy as np

from straxen.common import pax_file, get_resource, get_elife, first_sr1_run
from straxen.itp_map import InterpolatingMap
export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('trigger_min_area', default=100,
                 help='Peaks must have more area (PE) than this to '
                      'cause events'),
    strax.Option('trigger_max_competing', default=7,
                 help='Peaks must have FEWER nearby larger or slightly smaller'
                      ' peaks to cause events'),
    strax.Option('left_event_extension', default=int(1e6),
                 help='Extend events this many ns to the left from each '
                      'triggering peak'),
    strax.Option('right_event_extension', default=int(1e6),
                 help='Extend events this many ns to the right from each '
                      'triggering peak'),
)
class Events(strax.OverlapWindowPlugin):
    depends_on = ['peak_basics', 'peak_proximity']
    data_kind = 'events'
    dtype = [
        ('event_number', np.int64, 'Event number in this dataset'),
        ('time', np.int64, 'Event start time in ns since the unix epoch'),
        ('endtime', np.int64, 'Event end time in ns since the unix epoch')]
    events_seen = 0

    def get_window_size(self):
        # Take a large window for safety, events can have long tails
        return 10 * (self.config['left_event_extension']
                     + self.config['right_event_extension'])

    def compute(self, peaks, start, end):
        le = self.config['left_event_extension']
        re = self.config['right_event_extension']

        triggers = peaks[
            (peaks['area'] > self.config['trigger_min_area'])
            & (peaks['n_competing'] <= self.config['trigger_max_competing'])]

        # Join nearby triggers
        t0, t1 = strax.find_peak_groups(
            triggers,
            gap_threshold=le + re + 1,
            left_extension=le,
            right_extension=re)

        # Don't extend beyond the chunk boundaries
        # This will often happen for events near the invalid boundary of the
        # overlap processing (which should be thrown away)
        t0 = np.clip(t0, start, end)
        t1 = np.clip(t1, start, end)

        result = np.zeros(len(t0), self.dtype)
        result['time'] = t0
        result['endtime'] = t1
        result['event_number'] = np.arange(len(result)) + self.events_seen

        if not result.size > 0:
            print("Found chunk without events?!")

        self.events_seen += len(result)

        return result
        # TODO: someday investigate if/why loopplugin doesn't give
        # anything if events do not contain peaks..
        # Likely this has been resolved in 6a2cc6c


@export
class EventBasics(strax.LoopPlugin):
    __version__ = '0.3.0'
    depends_on = ('events',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')

    def infer_dtype(self):
        dtype = [(('Number of peaks in the event',
                   'n_peaks'), np.int32),
                 (('Drift time between main S1 and S2 in ns',
                   'drift_time'), np.int64)]
        for i in [1, 2]:
            dtype += [
                ((f'Main S{i} peak index',
                  f's{i}_index'), np.int32),
                ((f'Main S{i} time since unix epoch [ns]',
                  f's{i}_time'), np.int64),
                ((f'Main S{i} weighted center time since unix epoch [ns]',
                  f's{i}_center_time'), np.int64),
                ((f'Alternate S{i} time since unix epoch [ns]',
                  f'alt_s{i}_time'), np.int64),
                ((f'Alternate S{i} weighted center time since unix epoch [ns]',
                  f'alt_s{i}_center_time'), np.int64),
                ((f'Main S{i} area, uncorrected [PE]',
                  f's{i}_area'), np.float32),
                ((f'Main S{i} area fraction top',
                  f's{i}_area_fraction_top'), np.float32),
                ((f'Main S{i} width, 50% area [ns]',
                  f's{i}_range_50p_area'), np.float32),
                ((f'Main S{i} number of competing peaks',
                  f's{i}_n_competing'), np.int32),
                ((f'Area of alternate S{i} in event [PE]',
                  f'alt_s{i}_area'), np.float32),
                ((f'Drift time using alternate S{i} [ns]',
                  f'alt_s{i}_interaction_drift_time'), np.float32)]
        dtype += [('x_s2', np.float32,
                   'Main S2 reconstructed X position, uncorrected [cm]',),
                  ('y_s2', np.float32,
                   'Main S2 reconstructed Y position, uncorrected [cm]',)]
        dtype += strax.time_fields

        return dtype

    def compute_loop(self, event, peaks):
        result = dict(n_peaks=len(peaks),
                      time=event['time'],
                      endtime=strax.endtime(event))
        if not len(peaks):
            return result

        main_s = dict()
        secondary_s = dict()
        for s_i in [2, 1]:
            s_mask = peaks['type'] == s_i

            # For determining the main / alternate S1s,
            # remove all peaks after the main S2 (if there was one)
            # since these cannot be related to the main S2.
            # This is why S2 finding happened first.
            if s_i == 1 and result[f's2_index'] != -1:
                s_mask &= peaks['time'] < main_s[2]['time']

            ss = peaks[s_mask]
            s_indices = np.arange(len(peaks))[s_mask]

            if not len(ss):
                result[f's{s_i}_index'] = -1
                continue

            main_i = np.argmax(ss['area'])
            result[f's{s_i}_index'] = s_indices[main_i]

            if ss['n_competing'][main_i] > 0 and len(ss['area']) > 1:
                # Find second largest S..
                secondary_s[s_i] = x = ss[np.argsort(ss['area'])[-2]]
                for prop in ['area', 'time', 'center_time']:
                    result[f'alt_s{s_i}_{prop}'] = x[prop]

            s = main_s[s_i] = ss[main_i]
            for prop in ['area', 'area_fraction_top', 'time', 'center_time',
                         'range_50p_area', 'n_competing']:
                result[f's{s_i}_{prop}'] = s[prop]
            if s_i == 2:
                for q in 'xy':
                    result[f'{q}_s2'] = s[q]

        # Compute a drift time only if we have a valid S1-S2 pairs
        if len(main_s) == 2:
            result['drift_time'] = \
                main_s[2]['center_time'] - main_s[1]['center_time']
            if 1 in secondary_s:
                result['alt_s1_interaction_drift_time'] = \
                    main_s[2]['center_time'] - secondary_s[1]['center_time']
            if 2 in secondary_s:
                result['alt_s2_interaction_drift_time'] = \
                    secondary_s[2]['center_time'] - main_s[1]['center_time']

        return result


@export
@strax.takes_config(
    strax.Option(
        name='electron_drift_velocity',
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)',
        default=1.3325e-4
    ),
    strax.Option(
        'fdc_map',
        help='3D field distortion correction map path',
        default_by_run=[
            (0, pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz')),  # noqa
            (first_sr1_run, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part1_v1.json.gz')),  # noqa
            (170411_0611, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part2_v1.json.gz')),  # noqa
            (170704_0556, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part3_v1.json.gz')),  # noqa
            (170925_0622, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part4_v1.json.gz'))]),  # noqa
)
class EventPositions(strax.Plugin):
    __version__ = '0.1.0'

    depends_on = ('event_basics',)
    dtype = [
        ('x', np.float32,
         'Interaction x-position, field-distortion corrected (cm)'),
        ('y', np.float32,
         'Interaction y-position, field-distortion corrected (cm)'),
        ('z', np.float32,
         'Interaction z-position, field-distortion corrected (cm)'),
        ('r', np.float32,
         'Interaction radial position, field-distortion corrected (cm)'),
        ('z_naive', np.float32,
         'Interaction z-position using mean drift velocity only (cm)'),
        ('r_naive', np.float32,
         'Interaction r-position using observed S2 positions directly (cm)'),
        ('r_field_distortion_correction', np.float32,
         'Correction added to r_naive for field distortion (cm)'),
        ('theta', np.float32,
         'Interaction angular position (radians)')
    ] + strax.time_fields

    def setup(self):
        self.map = InterpolatingMap(
            get_resource(self.config['fdc_map'], fmt='binary'))

    def compute(self, events):
        z_obs = - self.config['electron_drift_velocity'] * events['drift_time']

        orig_pos = np.vstack([events['x_s2'], events['y_s2'], z_obs]).T
        r_obs = np.linalg.norm(orig_pos[:, :2], axis=1)

        delta_r = self.map(orig_pos)
        with np.errstate(invalid='ignore', divide='ignore'):
            r_cor = r_obs + delta_r
            scale = r_cor / r_obs

        result = dict(time=events['time'],
                      endtime=strax.endtime(events),
                      x=orig_pos[:, 0] * scale,
                      y=orig_pos[:, 1] * scale,
                      r=r_cor,
                      z_naive=z_obs,
                      r_naive=r_obs,
                      r_field_distortion_correction=delta_r,
                      theta=np.arctan2(orig_pos[:, 1], orig_pos[:, 0]))

        with np.errstate(invalid='ignore'):
            z_cor = -(z_obs ** 2 - delta_r ** 2) ** 0.5
            invalid = np.abs(z_obs) < np.abs(delta_r)        # Why??
        z_cor[invalid] = z_obs[invalid]
        result['z'] = z_cor

        return result


@strax.takes_config(
    strax.Option(
        's1_relative_lce_map',
        help="S1 relative LCE(x,y,z) map",
        default_by_run=[
            (0, pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR0_pax-680_fdc-3d_v0.json')),  # noqa
            (first_sr1_run, pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json'))]),  # noqa
    strax.Option(
        's2_relative_lce_map',
        help="S2 relative LCE(x, y) map",
        default_by_run=[
            (0, pax_file('XENON1T_s2_xy_ly_SR0_24Feb2017.json')),
            (170118_1327, pax_file('XENON1T_s2_xy_ly_SR1_v2.2.json'))]),
   strax.Option(
        'elife_file',
        default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/elife.npy',
        help='link to the electron lifetime'))
class CorrectedAreas(strax.Plugin):
    __version__ = '0.1.0'

    depends_on = ['event_basics', 'event_positions']
    dtype = [('cs1', np.float32, 'Corrected S1 area (PE)'),
             ('cs2', np.float32, 'Corrected S2 area (PE)')
             ] + strax.time_fields

    def setup(self):
        self.s1_map = InterpolatingMap(
            get_resource(self.config['s1_relative_lce_map']))
        self.s2_map = InterpolatingMap(
            get_resource(self.config['s2_relative_lce_map']))
        self.elife = get_elife(self.run_id,self.config['elife_file'])

    def compute(self, events):
        event_positions = np.vstack([events['x'], events['y'], events['z']]).T
        s2_positions = np.vstack([events['x_s2'], events['y_s2']]).T
        lifetime_corr = np.exp(
            events['drift_time'] / self.elife)

        return dict(
            time=events['time'],
            endtime=strax.endtime(events),
            cs1=events['s1_area'] / self.s1_map(event_positions),
            cs2=events['s2_area'] * lifetime_corr / self.s2_map(s2_positions))


@strax.takes_config(
    strax.Option(
        'g1',
        help="S1 gain in PE / photons produced",
        default_by_run=[(0, 0.1442),
                        (first_sr1_run, 0.1426)]),
    strax.Option(
        'g2',
        help="S2 gain in PE / electrons produced",
        default_by_run=[(0, 11.52/(1 - 0.63)),
                        (first_sr1_run, 11.55/(1 - 0.63))]),
    strax.Option(
        'lxe_w',
        help="LXe work function in quanta/keV",
        default=13.7e-3),
)
class EnergyEstimates(strax.Plugin):
    __version__ = '0.1.0'
    depends_on = ['corrected_areas']
    dtype = [
        ('e_light', np.float32, 'Energy in light signal [keVee]'),
        ('e_charge', np.float32, 'Energy in charge signal [keVee]'),
        ('e_ces', np.float32, 'Energy estimate [keVee]')
    ] + strax.time_fields

    def compute(self, events):
        el = self.cs1_to_e(events['cs1'])
        ec = self.cs2_to_e(events['cs2'])
        return dict(e_light=el,
                    e_charge=ec,
                    e_ces=el + ec,
                    time=events['time'],
                    endtime=strax.endtime(events))

    def cs1_to_e(self, x):
        return self.config['lxe_w'] * x / self.config['g1']

    def cs2_to_e(self, x):
        return self.config['lxe_w'] * x / self.config['g2']


class EventInfo(strax.MergeOnlyPlugin):
    depends_on = ['events',
                  'event_basics', 'event_positions', 'corrected_areas',
                  'energy_estimates']
    save_when = strax.SaveWhen.ALWAYS
