import strax
import numpy as np
from warnings import warn
from .position_reconstruction import DEFAULT_POSREC_ALGO_OPTION
from straxen.common import pax_file, get_resource, first_sr1_run
from straxen.get_corrections import get_correction_from_cmt, get_config_from_cmt, get_elife
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
    strax.Option('left_event_extension', default=int(2.7e6),
                 help='Extend events this many ns to the left from each '
                      'triggering peak'),
    strax.Option('right_event_extension', default=int(0.5e6),
                 help='Extend events this many ns to the right from each '
                      'triggering peak'),
)
class Events(strax.OverlapWindowPlugin):
    """
    Plugin which defines an "event" in our TPC.

    An event is defined by peak(s) in fixed range of time around a peak
    which satisfies certain conditions:
        1. The triggering peak must have a certain area.
        2. The triggering peak must have less than
        "trigger_max_competing" peaks. (A competing peak must have a
        certain area fraction of the triggering peak and must be in a
        window close to the main peak)

    Note:
        The time range which defines an event gets chopped at the chunk
        boundaries. This happens at invalid boundaries of the
    """
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
@strax.takes_config(
    strax.Option(
        name='allow_posts2_s1s', default=False,
        help="Allow S1s past the main S2 to become the main S1 and S2"),
    strax.Option(
        name='force_main_before_alt', default=False,
        help="Make the alternate S1 (and likewise S2) the main S1 if "
             "occurs before the main S1.")
)
class EventBasics(strax.LoopPlugin):
    """
    Computes the basic properties of the main/alternative S1/S2 within
    an event.

    The main S2 and alternative S2 are given by the largest two S2-Peaks
    within the event. By default this is also true for S1.
    """
    __version__ = '0.5.7'

    depends_on = ('events',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')
    provides = ('event_basics', 'event_posrec_many')
    data_kind = {k: 'events' for k in provides}
    loop_over = 'events'

    # Properties to store for each peak (main and alternate S1 and S2)
    peak_properties = (
        # name                dtype       comment
        ('time',              np.int64,   'start time since unix epoch [ns]'),
        ('center_time',       np.int64,   'weighted center time since unix epoch [ns]'),
        ('endtime',           np.int64,   'end time since unix epoch [ns]'),
        ('area',              np.float32, 'area, uncorrected [PE]'),
        ('n_channels',        np.int32,   'count of contributing PMTs'),
        ('n_competing',       np.float32, 'number of competing PMTs'),
        ('range_50p_area',    np.float32, 'width, 50% area [ns]'),
        ('area_fraction_top', np.float32, 'fraction of area seen by the top PMT array'))

    def infer_dtype(self):
        # Basic event properties
        basics_dtype = []
        basics_dtype += strax.time_fields
        basics_dtype += [('n_peaks', np.int32, 'Number of peaks in the event'),
                         ('drift_time', np.int32,
                          'Drift time between main S1 and S2 in ns')]

        for i in [1, 2]:
            # Peak indices
            basics_dtype += [
                (f's{i}_index', np.int32,
                 f'Main S{i} peak index in event'),
                (f'alt_s{i}_index', np.int32,
                 f'Alternate S{i} peak index in event')]

            # Peak properties
            for name, dt, comment in self.peak_properties:
                basics_dtype += [
                    (f's{i}_{name}', dt, f'Main S{i} {comment}'),
                    (f'alt_s{i}_{name}', dt, f'Alternate S{i} {comment}')]

            # Drifts and delays
            basics_dtype += [
                (f'alt_s{i}_interaction_drift_time', np.int32,
                 f'Drift time using alternate S{i} [ns]'),
                (f'alt_s{i}_delay', np.int32,
                 f'Time between main and alternate S{i} [ns]')]

        basics_dtype += [
            (f's2_x', np.float32,
             f'Main S2 reconstructed X position, uncorrected [cm]'),
            (f's2_y', np.float32,
             f'Main S2 reconstructed Y position, uncorrected [cm]'),
            (f'alt_s2_x', np.float32,
             f'Alternate S2 reconstructed X position, uncorrected [cm]'),
            (f'alt_s2_y', np.float32,
             f'Alternate S2 reconstructed Y position, uncorrected [cm]')]

        posrec_many_dtype = list(strax.time_fields)
        # parse x_mlp et cetera if needed to get the algorithms used.
        self.pos_rec_labels = list(
            set(d.split('_')[-1] for d in
                self.deps['peak_positions'].dtype_for('peak_positions').names
                if 'x_' in d))
        # Preserve order. "set" is not ordered and dtypes should always be ordered
        self.pos_rec_labels.sort()

        for algo in self.pos_rec_labels:
            # S2 positions
            posrec_many_dtype += [
                (f's2_x_{algo}', np.float32,
                 f'Main S2 {algo}-reconstructed X position, uncorrected [cm]'),
                (f's2_y_{algo}', np.float32,
                 f'Main S2 {algo}-reconstructed Y position, uncorrected [cm]'),
                (f'alt_s2_x_{algo}', np.float32,
                 f'Alternate S2 {algo}-reconstructed X position, uncorrected [cm]'),
                (f'alt_s2_y_{algo}', np.float32,
                 f'Alternate S2 {algo}-reconstructed Y position, uncorrected [cm]')]

        return {'event_basics': basics_dtype,
                'event_posrec_many': posrec_many_dtype}

    def compute_loop(self, event, peaks):
        result = dict(n_peaks=len(peaks),
                      time=event['time'],
                      endtime=strax.endtime(event))
        posrec_result = dict(time=event['time'],
                             endtime=strax.endtime(event))
        posrec_save = [d.replace("s2_", "").replace("alt_", "")
                       for d in self.dtype_for('event_posrec_many').names if
                       'time' not in d]

        if not len(peaks):
            return result
        main_s = dict()
        secondary_s = dict()

        # Consider S2s first, then S1s (to enable allow_posts2_s1s = False)
        for s_i in [2, 1]:

            # Which properties do we need?
            to_store = [name for name, _, _ in self.peak_properties]
            if s_i == 2:
                to_store += ['x', 'y']

            # Find all peaks of this type (S1 or S2)
            s_mask = peaks['type'] == s_i
            if not self.config['allow_posts2_s1s']:
                # Only peaks *before* the main S2 are allowed to be
                # the main or alternate S1
                if s_i == 1 and result[f's2_index'] != -1:
                    s_mask &= peaks['time'] < main_s[2]['time']
            ss = peaks[s_mask]
            s_indices = np.arange(len(peaks))[s_mask]

            # Decide which of these signals is the main and alternate
            if len(ss) > 1:
                # Start by choosing the largest two signals
                _alt_i, _main_i = np.argsort(ss['area'])[-2:]
                if (self.config['force_main_before_alt']
                        and ss[_alt_i]['time'] < ss[_main_i]['time']):
                    # Promote alternate to main since it occurs earlier
                    _alt_i, _main_i = _main_i, _alt_i
            elif len(ss) == 1:
                _main_i, _alt_i = 0, None
            else:
                _alt_i, _main_i = None, None

            # Store main signal properties
            if _main_i is None:
                result[f's{s_i}_index'] = -1
            else:
                main_s[s_i] = ss[_main_i]
                result[f's{s_i}_index'] = s_indices[_main_i]
                for name in to_store:
                    result[f's{s_i}_{name}'] = main_s[s_i][name]
                if s_i == 2:
                    for name in posrec_save:
                        posrec_result[f's{s_i}_{name}'] = main_s[s_i][name]
                        
            # Store alternate signal properties
            if _alt_i is None:
                result[f'alt_s{s_i}_index'] = -1
            else:
                secondary_s[s_i] = ss[_alt_i]
                result[f'alt_s{s_i}_index'] = s_indices[_alt_i]
                for name in to_store:
                    result[f'alt_s{s_i}_{name}'] = secondary_s[s_i][name]
                if s_i == 2:
                    for name in posrec_save:
                        posrec_result[f'alt_s{s_i}_{name}'] = secondary_s[s_i][name]
                # Compute delay time properties
                result[f'alt_s{s_i}_delay'] = (secondary_s[s_i]['center_time']
                                               - main_s[s_i]['center_time'])

        # Compute drift times only if we have a valid S1-S2 pair
        if len(main_s) == 2:
            result['drift_time'] = \
                main_s[2]['center_time'] - main_s[1]['center_time']
            if 1 in secondary_s:
                result['alt_s1_interaction_drift_time'] = \
                    main_s[2]['center_time'] - secondary_s[1]['center_time']
            if 2 in secondary_s:
                result['alt_s2_interaction_drift_time'] = \
                    secondary_s[2]['center_time'] - main_s[1]['center_time']

        return {'event_basics': result,
                'event_posrec_many': posrec_result}


@export
@strax.takes_config(
    strax.Option(
        name='electron_drift_velocity',
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)',
        default=1.3325e-4
    ),
    strax.Option(
        name='fdc_map',
        help='3D field distortion correction map path',
        default_by_run=[
            (0, pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz')),  # noqa
            (first_sr1_run, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part1_v1.json.gz')), # noqa
            (170411_0611, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part2_v1.json.gz')), # noqa
            (170704_0556, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part3_v1.json.gz')), # noqa
            (170925_0622, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part4_v1.json.gz'))], # noqa
    ),
    *DEFAULT_POSREC_ALGO_OPTION
)
class EventPositions(strax.Plugin):
    """
    Computes the observed and corrected position for the main S1/S2
    pairs in an event. For XENONnT data, it returns the FDC corrected
    positions of the default_reconstruction_algorithm. In case the fdc_map
    is given as a file (not through CMT), then the coordinate system
    should be given as (x, y, z), not (x, y, drift_time).
    """

    depends_on = ('event_basics', )
    
    __version__ = '0.1.3'

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

        is_CMT = isinstance(self.config['fdc_map'], tuple)

        if is_CMT:

            cmt, cmt_conf, is_nt = self.config['fdc_map']
            cmt_conf = (f'{cmt_conf[0]}_{self.config["default_reconstruction_algorithm"]}' , cmt_conf[1])
            map_algo = cmt, cmt_conf, is_nt           
 
            self.map = InterpolatingMap(
                get_resource(get_config_from_cmt(self.run_id, map_algo), fmt='binary'))
            self.map.scale_coordinates([1., 1., -self.config['electron_drift_velocity']])

        elif isinstance(self.config['fdc_map'], str):
            self.map = InterpolatingMap(
                get_resource(self.config['fdc_map'], fmt='binary'))

        else:
            raise NotImplementedError('FDC map format not understood.')

    def compute(self, events):

        result = {'time': events['time'],
                  'endtime': strax.endtime(events)}
        
        z_obs = - self.config['electron_drift_velocity'] * events['drift_time']
        orig_pos = np.vstack([events[f's2_x'], events[f's2_y'], z_obs]).T
        r_obs = np.linalg.norm(orig_pos[:, :2], axis=1)
        delta_r = self.map(orig_pos)

        # apply radial correction
        with np.errstate(invalid='ignore', divide='ignore'):
            r_cor = r_obs + delta_r
            scale = r_cor / r_obs

        # z correction due to longer drift time for distortion
        # (geometrical reasoning not valid if |delta_r| > |z_obs|,
        #  as cathetus cannot be longer than hypothenuse)
        with np.errstate(invalid='ignore'):
            z_cor = -(z_obs ** 2 - delta_r ** 2) ** 0.5
            invalid = np.abs(z_obs) < np.abs(delta_r)
        z_cor[invalid] = z_obs[invalid]

        result.update({'x': orig_pos[:, 0] * scale,
                       'y': orig_pos[:, 1] * scale,
                       'r': r_cor,
                       'r_naive': r_obs,
                       'r_field_distortion_correction': delta_r,
                       'theta': np.arctan2(orig_pos[:, 1], orig_pos[:, 0]),
                       'z_naive': z_obs,
                       'z': z_cor})

        return result


@strax.takes_config(
    strax.Option(
        's1_relative_lce_map',
        help="S1 relative LCE(x,y,z) map",
        default_by_run=[
            (0, pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR0_pax-680_fdc-3d_v0.json')),  # noqa
            (first_sr1_run, pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json'))]),  # noqa
    strax.Option(
        's2_xy_correction_map',
        help="S2 (x, y) correction map. Correct S2 position dependence "
             "manly due to bending of anode/gate-grid, PMT quantum efficiency "
             "and extraction field distribution, as well as other geometric factors.",
        default_by_run=[
            (0, pax_file('XENON1T_s2_xy_ly_SR0_24Feb2017.json')),
            (170118_1327, pax_file('XENON1T_s2_xy_ly_SR1_v2.2.json'))]),
   strax.Option(
        'elife_conf',
        default=("elife", "ONLINE", True),
        help='Electron lifetime '
             'Specify as (model_type->str, model_config->str, is_nT->bool) '
             'where model_type can be "elife" or "elife_constant" '
             'and model_config can be a version.'
   ))
class CorrectedAreas(strax.Plugin):
    """
    Plugin which applies light collection efficiency maps and electron
    life time to the data.

    Computes the cS1/cS2 for the main/alternative S1/S2 as well as the
    corrected life time.

    Note:
        Please be aware that for both, the main and alternative S1, the
        area is corrected according to the xy-position of the main S2.
    """
    __version__ = '0.1.0'

    depends_on = ['event_basics', 'event_positions']
    dtype = [('cs1', np.float32, 'Corrected S1 area [PE]'),
             ('cs2', np.float32, 'Corrected S2 area [PE]'),
             ('alt_cs1', np.float32, 'Corrected area of the alternate S1 [PE]'),
             ('alt_cs2', np.float32, 'Corrected area of the alternate S2 [PE]')
             ] + strax.time_fields

    def setup(self):

        self.s1_map = InterpolatingMap(
                get_resource(self.config['s1_relative_lce_map']))
        self.s2_map = InterpolatingMap(
                get_resource(get_config_from_cmt(self.run_id, self.config['s2_xy_correction_map'])))
        self.elife = get_correction_from_cmt(self.run_id, self.config['elife_conf'])

        if isinstance(self.elife, str):
            # Legacy 1T support
            self.elife = get_elife(self.run_id, self.elife)

    def compute(self, events):
        # S1 corrections depend on the actual corrected event position.
        # We use this also for the alternate S1; for e.g. Kr this is
        # fine as the S1 correction varies slowly.
        event_positions = np.vstack([events['x'], events['y'], events['z']]).T

        # For electron lifetime corrections to the S2s,
        # use lifetimes computed using the main S1.
        lifetime_corr = np.exp(events['drift_time'] / self.elife)
        alt_lifetime_corr = (
            np.exp((events['alt_s2_interaction_drift_time'])
                   / self.elife))

        # S2(x,y) corrections use the observed S2 positions
        s2_positions = np.vstack([events['s2_x'], events['s2_y']]).T
        alt_s2_positions = np.vstack([events['alt_s2_x'], events['alt_s2_y']]).T

        return dict(
            time=events['time'],
            endtime=strax.endtime(events),

            cs1=events['s1_area'] / self.s1_map(event_positions),
            alt_cs1=events['alt_s1_area'] / self.s1_map(event_positions),

            cs2=(events['s2_area'] * lifetime_corr
                 / self.s2_map(s2_positions)),
            alt_cs2=(events['alt_s2_area'] * alt_lifetime_corr
                     / self.s2_map(alt_s2_positions)))


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
    """
    Plugin which converts cS1 and cS2 into energies (from PE to KeVee).
    """
    __version__ = '0.1.0'
    depends_on = ['corrected_areas']
    dtype = [
        ('e_light', np.float32, 'Energy in light signal [keVee]'),
        ('e_charge', np.float32, 'Energy in charge signal [keVee]'),
        ('e_ces', np.float32, 'Energy estimate [keVee]')
    ] + strax.time_fields
    save_when = strax.SaveWhen.TARGET

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
    """
    Plugin which merges the information of all event data_kinds into a
    single data_type.
    """
    depends_on = ['events',
                  'event_basics', 'event_positions', 'corrected_areas',
                  'energy_estimates']
    save_when = strax.SaveWhen.ALWAYS
