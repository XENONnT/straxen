import strax
import numpy as np
import numba
import straxen
from warnings import warn
from .position_reconstruction import DEFAULT_POSREC_ALGO_OPTION
from straxen.common import pax_file, get_resource, first_sr1_run, pre_apply_function
from straxen.get_corrections import get_correction_from_cmt, get_cmt_resource, is_cmt_option
from straxen.itp_map import InterpolatingMap
export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('trigger_min_area', default=100, type=(int,float),
                 help='Peaks must have more area (PE) than this to '
                      'cause events'),
    strax.Option('trigger_max_competing', default=7, type=int,
                 help='Peaks must have FEWER nearby larger or slightly smaller'
                      ' peaks to cause events'),
    strax.Option('left_event_extension', default=int(0.25e6), type=(int, float),
                 help='Extend events this many ns to the left from each '
                      'triggering peak. This extension is added to the maximum '
                      'drift time.',
                 ),
    strax.Option('right_event_extension', default=int(0.25e6), type=(int, float),
                 help='Extend events this many ns to the right from each '
                      'triggering peak.',
                 ),
    strax.Option(name='electron_drift_velocity', infer_type=False,
                 default=("electron_drift_velocity", "ONLINE", True),
                 help='Vertical electron drift velocity in cm/ns (1e4 m/ms)',
                 ),
    strax.Option(name='max_drift_length',
                 default=straxen.tpc_z, type=(int, float),
                 help='Total length of the TPC from the bottom of gate to the '
                      'top of cathode wires [cm]',
                 ),
    strax.Option(name='exclude_s1_as_triggering_peaks',
                 default=True, type=bool,
                 help='If true exclude S1s as triggering peaks.',
                 ),
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
    provides = 'events'
    data_kind = 'events'
    __version__ = '0.1.0'
    save_when = strax.SaveWhen.NEVER

    dtype = [
        ('event_number', np.int64, 'Event number in this dataset'),
        ('time', np.int64, 'Event start time in ns since the unix epoch'),
        ('endtime', np.int64, 'Event end time in ns since the unix epoch')]

    events_seen = 0

    def setup(self):
        electron_drift_velocity = get_correction_from_cmt(
            self.run_id,
            self.config['electron_drift_velocity'])
        self.drift_time_max = int(self.config['max_drift_length'] / electron_drift_velocity)
        # Left_extension and right_extension should be computed in setup to be
        # reflected in cutax too.
        self.left_extension = self.config['left_event_extension'] + self.drift_time_max
        self.right_extension = self.config['right_event_extension']

    def get_window_size(self):
        # Take a large window for safety, events can have long tails
        return 10 * (self.config['left_event_extension']
                     + self.drift_time_max
                     + self.config['right_event_extension'])

    def compute(self, peaks, start, end):
        le = self.config['left_event_extension'] + self.drift_time_max
        re = self.config['right_event_extension']

        _is_triggering = peaks['area'] > self.config['trigger_min_area']
        _is_triggering &= (peaks['n_competing'] <= self.config['trigger_max_competing'])
        if self.config['exclude_s1_as_triggering_peaks']:
            _is_triggering &= peaks['type'] == 2

        triggers = peaks[_is_triggering]

        # Join nearby triggers
        t0, t1 = strax.find_peak_groups(
            triggers,
            gap_threshold=self.left_extension + self.right_extension + 1,
            left_extension=self.left_extension,
            right_extension=self.right_extension)

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


@export
@strax.takes_config(
    strax.Option(
        name='allow_posts2_s1s', default=False, infer_type=False,
        help="Allow S1s past the main S2 to become the main S1 and S2"),
    strax.Option(
        name='force_main_before_alt', default=False, infer_type=False,
        help="Make the alternate S1 (and likewise S2) the main S1 if "
             "occurs before the main S1."),
    strax.Option(
        name='event_s1_min_coincidence',
        default=2, infer_type=False,
        help="Event level S1 min coincidence. Should be >= s1_min_coincidence "
             "in the peaklet classification"),
)
class EventBasics(strax.Plugin):
    """
    Computes the basic properties of the main/alternative S1/S2 within
    an event.

    The main S2 and alternative S2 are given by the largest two S2-Peaks
    within the event. By default this is also true for S1.
    """
    __version__ = '1.2.1'

    depends_on = ('events',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')
    provides = 'event_basics'
    data_kind = 'events'
    loop_over = 'events'

    def infer_dtype(self):
        # Basic event properties
        self._set_posrec_save()
        self._set_dtype_requirements()
        dtype = []
        dtype += strax.time_fields
        dtype += [('n_peaks', np.int32,
                   'Number of peaks in the event'),
                  ('drift_time', np.float32,
                   'Drift time between main S1 and S2 in ns'),
                  ('event_number', np.int64,
                   'Event number in this dataset'),
                  ]

        dtype += self._get_si_dtypes(self.peak_properties)

        dtype += [
            (f's2_x', np.float32,
             f'Main S2 reconstructed X position, uncorrected [cm]'),
            (f's2_y', np.float32,
             f'Main S2 reconstructed Y position, uncorrected [cm]'),
            (f'alt_s2_x', np.float32,
             f'Alternate S2 reconstructed X position, uncorrected [cm]'),
            (f'alt_s2_y', np.float32,
             f'Alternate S2 reconstructed Y position, uncorrected [cm]'),
            (f'area_before_main_s2', np.float32,
             f'Sum of areas before Main S2 [PE]'),
            (f'large_s2_before_main_s2', np.float32,
             f'The largest S2 before the Main S2 [PE]')
        ]

        dtype += self._get_posrec_dtypes()
        return dtype

    def _set_dtype_requirements(self):
        """Needs to be run before inferring dtype as it is needed there"""
        # Properties to store for each peak (main and alternate S1 and S2)
        self.peak_properties = (
            # name                dtype       comment
            ('time',              np.int64,   'start time since unix epoch [ns]'),
            ('center_time',       np.int64,   'weighted center time since unix epoch [ns]'),
            ('endtime',           np.int64,   'end time since unix epoch [ns]'),
            ('area',              np.float32, 'area, uncorrected [PE]'),
            ('n_channels',        np.int16,   'count of contributing PMTs'),
            ('n_competing',       np.int32,   'number of competing peaks'),
            ('max_pmt',           np.int16,   'PMT number which contributes the most PE'),
            ('max_pmt_area',      np.float32, 'area in the largest-contributing PMT (PE)'),
            ('range_50p_area',    np.float32, 'width, 50% area [ns]'),
            ('range_90p_area',    np.float32, 'width, 90% area [ns]'),
            ('rise_time',         np.float32, 'time between 10% and 50% area quantiles [ns]'),
            ('area_fraction_top', np.float32, 'fraction of area seen by the top PMT array'),
            ('tight_coincidence', np.int16, 'Channel within tight range of mean'),
            ('n_saturated_channels',      np.int16, 'Total number of saturated channels'),
        )

    @staticmethod
    def _get_si_dtypes(peak_properties):
        """Get properties for S1/S2 from peaks directly"""
        si_dtype = []
        for s_i in [1, 2]:
            # Peak indices
            si_dtype += [
                (f's{s_i}_index', np.int32, f'Main S{s_i} peak index in event'),
                (f'alt_s{s_i}_index', np.int32, f'Alternate S{s_i} peak index in event')]

            # Peak properties
            for name, dt, comment in peak_properties:
                si_dtype += [(f's{s_i}_{name}', dt, f'Main S{s_i} {comment}'),
                             (f'alt_s{s_i}_{name}', dt, f'Alternate S{s_i} {comment}')]

            # Drifts and delays
            si_dtype += [
                (f'alt_s{s_i}_interaction_drift_time', np.float32,
                 f'Drift time using alternate S{s_i} [ns]'),
                (f'alt_s{s_i}_delay', np.int32,
                 f'Time between main and alternate S{s_i} [ns]')]
        return si_dtype

    def _set_posrec_save(self):
        """
        parse x_mlp et cetera if needed to get the algorithms used and
        set required class attributes
        """
        posrec_fields = self.deps['peak_positions'].dtype_for('peak_positions').names
        posrec_names = [d.split('_')[-1] for d in posrec_fields if 'x_' in d]

        # Preserve order. "set" is not ordered and dtypes should always be ordered
        self.pos_rec_labels = list(set(posrec_names))
        self.pos_rec_labels.sort()

        self.posrec_save = [(xy + algo)
                            for xy in ['x_', 'y_']
                            for algo in self.pos_rec_labels]

    def _get_posrec_dtypes(self):
        """Get S2 positions for each of the position reconstruction algorithms"""
        posrec_dtpye = []

        for algo in self.pos_rec_labels:
            # S2 positions
            posrec_dtpye += [
                (f's2_x_{algo}', np.float32,
                 f'Main S2 {algo}-reconstructed X position, uncorrected [cm]'),
                (f's2_y_{algo}', np.float32,
                 f'Main S2 {algo}-reconstructed Y position, uncorrected [cm]'),
                (f'alt_s2_x_{algo}', np.float32,
                 f'Alternate S2 {algo}-reconstructed X position, uncorrected [cm]'),
                (f'alt_s2_y_{algo}', np.float32,
                 f'Alternate S2 {algo}-reconstructed Y position, uncorrected [cm]')]

        return posrec_dtpye

    @staticmethod
    def set_nan_defaults(buffer):
        """
        When constructing the dtype, take extra care to set values to
        np.Nan / -1 (for ints) as 0 might have a meaning
        """
        for field in buffer.dtype.names:
            if np.issubdtype(buffer.dtype[field], np.integer):
                buffer[field][:] = -1
            else:
                buffer[field][:] = np.nan

    def compute(self, events, peaks):
        result = np.zeros(len(events), dtype=self.dtype)
        self.set_nan_defaults(result)

        split_peaks = strax.split_by_containment(peaks, events)

        result['time'] = events['time']
        result['endtime'] = events['endtime']
        result['event_number'] = events['event_number']

        self.fill_events(result, events, split_peaks)
        return result

    # If copy_largest_peaks_into_event is ever numbafied, also numbafy this function
    def fill_events(self, result_buffer, events, split_peaks):
        """Loop over the events and peaks within that event"""
        for event_i, _ in enumerate(events):
            peaks_in_event_i = split_peaks[event_i]
            n_peaks = len(peaks_in_event_i)
            result_buffer[event_i]['n_peaks'] = n_peaks

            if not n_peaks:
                raise ValueError(f'No peaks within event?\n{events[event_i]}')

            self.fill_result_i(result_buffer[event_i], peaks_in_event_i)

    def fill_result_i(self, event, peaks):
        """For a single event with the result_buffer"""
        # Consider S2s first, then S1s (to enable allow_posts2_s1s = False)
        largest_s2s, s2_idx = self.get_largest_sx_peaks(peaks, s_i=2)

        if self.config['force_main_before_alt']:
            s2_order = np.argsort(largest_s2s['time'])
            largest_s2s = largest_s2s[s2_order]
            s2_idx = s2_idx[s2_order]

        if not self.config['allow_posts2_s1s'] and len(largest_s2s):
            s1_latest_time = largest_s2s[0]['time']
        else:
            s1_latest_time = np.inf

        largest_s1s, s1_idx = self.get_largest_sx_peaks(
            peaks,
            s_i=1,
            s1_before_time=s1_latest_time,
            s1_min_coincidence=self.config['event_s1_min_coincidence'])

        self.set_sx_index(event, s1_idx, s2_idx)
        self.set_event_properties(event, largest_s1s, largest_s2s, peaks)

        # Loop over S1s and S2s and over main / alt.
        for s_i, largest_s_i in enumerate([largest_s1s, largest_s2s], 1):
            # Largest index 0 -> main sx, 1 -> alt sx
            for largest_index, main_or_alt in enumerate(['s', 'alt_s']):
                peak_properties_to_save = [name for name, _, _ in self.peak_properties]
                if s_i == 2:
                    peak_properties_to_save += ['x', 'y']
                    peak_properties_to_save += self.posrec_save
                field_names = [f'{main_or_alt}{s_i}_{name}' for name in peak_properties_to_save]
                self.copy_largest_peaks_into_event(event,
                                                   largest_s_i,
                                                   largest_index,
                                                   field_names,
                                                   peak_properties_to_save)

    @staticmethod
    @numba.njit
    def set_event_properties(result, largest_s1s, largest_s2s, peaks):
        """Get properties like drift time and area before main S2"""
        # Compute drift times only if we have a valid S1-S2 pair
        if len(largest_s1s) > 0 and len(largest_s2s) > 0:
            result['drift_time'] = largest_s2s[0]['center_time'] - largest_s1s[0]['center_time']
            if len(largest_s1s) > 1:
                result['alt_s1_interaction_drift_time'] = largest_s2s[0]['center_time'] - largest_s1s[1]['center_time']
                result['alt_s1_delay'] = largest_s1s[1]['center_time'] - largest_s1s[0]['center_time']
            if len(largest_s2s) > 1:
                result['alt_s2_interaction_drift_time'] = largest_s2s[1]['center_time'] - largest_s1s[0]['center_time']
                result['alt_s2_delay'] = largest_s2s[1]['center_time'] - largest_s2s[0]['center_time']

        # areas before main S2
        if len(largest_s2s):
            peaks_before_ms2 = peaks[peaks['time'] < largest_s2s[0]['time']]
            result['area_before_main_s2'] = np.sum(peaks_before_ms2['area'])

            s2peaks_before_ms2 = peaks_before_ms2[peaks_before_ms2['type'] == 2]
            if len(s2peaks_before_ms2) == 0:
                result['large_s2_before_main_s2'] = 0
            else:
                result['large_s2_before_main_s2'] = np.max(s2peaks_before_ms2['area'])
        return result

    @staticmethod
    # @numba.njit <- works but slows if fill_events is not numbafied
    def get_largest_sx_peaks(peaks,
                             s_i,
                             s1_before_time=np.inf,
                             s1_min_coincidence=0,
                             number_of_peaks=2):
        """Get the largest S1/S2. For S1s allow a min coincidence and max time"""
        # Find all peaks of this type (S1 or S2)
        s_mask = peaks['type'] == s_i
        if s_i == 1:
            s_mask &= peaks['time'] < s1_before_time
            s_mask &= peaks['tight_coincidence'] >= s1_min_coincidence

        selected_peaks = peaks[s_mask]
        s_index = np.arange(len(peaks))[s_mask]
        largest_peaks = np.argsort(selected_peaks['area'])[-number_of_peaks:][::-1]
        return selected_peaks[largest_peaks], s_index[largest_peaks]

    # If only we could numbafy this... Unfortunatly we cannot.
    # Perhaps we could one day consider doing something like strax.copy_to_buffer
    @staticmethod
    def copy_largest_peaks_into_event(result,
                                      largest_s_i,
                                      main_or_alt_index,
                                      result_fields,
                                      peak_fields,
                                      ):
        """
        For one event, write all the peak_fields (e.g. "area") of the peak
        (largest_s_i) into their associated field in the event (e.g. s1_area),
        main_or_alt_index differentiates between main (index 0) and alt (index 1)
        """
        index_not_in_list_of_largest_peaks = main_or_alt_index >= len(largest_s_i)
        if index_not_in_list_of_largest_peaks:
            # There is no such peak. E.g. main_or_alt_index == 1 but largest_s_i = ["Main S1"]
            # Asking for index 1 doesn't work on a len 1 list of peaks.
            return

        for i, ev_field in enumerate(result_fields):
            p_field = peak_fields[i]
            if p_field not in ev_field:
                raise ValueError("Event fields must derive from the peak fields")
            result[ev_field] = largest_s_i[main_or_alt_index][p_field]

    @staticmethod
    # @numba.njit <- works but slows if fill_events is not numbafied
    def set_sx_index(res, s1_idx, s2_idx):
        if len(s1_idx):
            res['s1_index'] = s1_idx[0]
            if len(s1_idx) > 1:
                res['alt_s1_index'] = s1_idx[1]
        if len(s2_idx):
            res['s2_index'] = s2_idx[0]
            if len(s2_idx) > 1:
                res['alt_s2_index'] = s2_idx[1]


@export
@strax.takes_config(
    strax.Option(
        name='electron_drift_velocity', infer_type=False,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)',
        default=("electron_drift_velocity", "ONLINE", True)
    ),
    strax.Option(
        name='electron_drift_time_gate', infer_type=False,
        help='Electron drift time from the gate in ns',
        default=("electron_drift_time_gate", "ONLINE", True)
    ),
    strax.Option(
        name='fdc_map', infer_type=False,
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
    
    __version__ = '0.1.4'

    dtype = [
        ('x', np.float32,
         'Interaction x-position, field-distortion corrected (cm)'),
        ('y', np.float32,
         'Interaction y-position, field-distortion corrected (cm)'),
        ('z', np.float32,
         'Interaction z-position, using mean drift velocity only (cm)'),
        ('r', np.float32,
         'Interaction radial position, field-distortion corrected (cm)'),
        ('z_naive', np.float32,
         'Interaction z-position using mean drift velocity only (cm)'),
        ('r_naive', np.float32,
         'Interaction r-position using observed S2 positions directly (cm)'),
        ('r_field_distortion_correction', np.float32,
         'Correction added to r_naive for field distortion (cm)'),
        ('z_field_distortion_correction', np.float32,
         'Correction added to z_naive for field distortion (cm)'),
        ('theta', np.float32,
         'Interaction angular position (radians)')
            ] + strax.time_fields

    def setup(self):

        self.electron_drift_velocity = get_correction_from_cmt(self.run_id, self.config['electron_drift_velocity'])
        self.electron_drift_time_gate = get_correction_from_cmt(self.run_id, self.config['electron_drift_time_gate'])
        
        if isinstance(self.config['fdc_map'], str):
            self.map = InterpolatingMap(
                get_resource(self.config['fdc_map'], fmt='binary'))

        elif is_cmt_option(self.config['fdc_map']):
            self.map = InterpolatingMap(
                get_cmt_resource(self.run_id,
                                 tuple(['suffix',
                                        self.config['default_reconstruction_algorithm'],
                                        *self.config['fdc_map']]),
                                 fmt='binary'))
            self.map.scale_coordinates([1., 1., - self.electron_drift_velocity])

        else:
            raise NotImplementedError('FDC map format not understood.')

    def compute(self, events):

        result = {'time': events['time'],
                  'endtime': strax.endtime(events)}
        
        z_obs = - self.electron_drift_velocity * (events['drift_time'] - self.electron_drift_time_gate)
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
            # do not apply z correction above gate
            invalid |= z_obs >= 0
        z_cor[invalid] = z_obs[invalid]
        delta_z = z_cor - z_obs

        result.update({'x': orig_pos[:, 0] * scale,
                       'y': orig_pos[:, 1] * scale,
                       'r': r_cor,
                       'r_naive': r_obs,
                       'r_field_distortion_correction': delta_r,
                       'theta': np.arctan2(orig_pos[:, 1], orig_pos[:, 0]),
                       'z_naive': z_obs,
                       # using z_obs in agreement with the dtype description
                       # the FDC for z (z_cor) is found to be not reliable (see #527)
                       'z': z_obs,
                       'z_field_distortion_correction': delta_z
                       })

        return result


@export
class EventInfoVetos(strax.Plugin):
    """
    Plugin which combines event_info with the tagged peaks information
    from muon- and neutron-veto.
    """
    __version__ = '0.0.0'
    depends_on = ('event_basics', 'peak_veto_tags')
    provides = 'events_tagged'
    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for s_i in [1, 2]:
            for peak_type in ['', 'alt_']:
                dtype += [((f"Veto tag for {peak_type}S{s_i}: unatagged: 0, nveto: 1, mveto: 2, both: 3",
                            f"{peak_type}s{s_i}_veto_tag"), np.int8),
                          ((f"Time to closest veto interval for {peak_type}s{s_i}",
                            f"{peak_type}s{s_i}_dt_veto"), np.int64),
                          ]
        dtype += [(('Number of peaks tagged by NV/MV inside event', 'n_tagged_peaks'), np.int16)]
        return dtype

    def compute(self, events, peaks):
        split_tags = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']
        get_veto_tags(events, split_tags, result)

        return result


def get_veto_tags(events, split_tags, result):
    """
    Loops over events and tag main/alt S1/2 according to peak tag.

    :param events: Event_info data type to be tagged.
    :param split_tags: Tags split by events.
    """
    for tags_i, event_i, result_i in zip(split_tags, events, result):
        result_i['n_tagged_peaks'] = np.sum(tags_i['veto_tag'] > 0)
        for s_i in [1, 2]:
            for peak_type in ['', 'alt_']:
                if event_i[f'{peak_type}s{s_i}_index'] == -1:
                    continue

                index = event_i[f'{peak_type}s{s_i}_index']
                result_i[f'{peak_type}s{s_i}_veto_tag'] = tags_i[index]['veto_tag']
                result_i[f'{peak_type}s{s_i}_dt_veto'] = tags_i[index]['time_to_closest_veto']


@export
@strax.takes_config(
    strax.Option(
        's1_xyz_correction_map', infer_type=False,
        help="S1 relative (x, y, z) correction map",
        default_by_run=[
            (0, pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR0_pax-680_fdc-3d_v0.json')),  # noqa
            (first_sr1_run, pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json'))]),  # noqa
    strax.Option(
        's2_xy_correction_map', infer_type=False,
        help="S2 (x, y) correction map. Correct S2 position dependence, including S2 top, bottom, and total."
             "manly due to bending of anode/gate-grid, PMT quantum efficiency "
             "and extraction field distribution, as well as other geometric factors.",
        default_by_run=[
            (0, pax_file('XENON1T_s2_xy_ly_SR0_24Feb2017.json')),
            (170118_1327, pax_file('XENON1T_s2_xy_ly_SR1_v2.2.json'))]),
    strax.Option(
        'elife_conf', infer_type=False,
        default=("elife", "ONLINE", True),
        help='Electron lifetime '
             'Specify as (model_type->str, model_config->str, is_nT->bool) '
             'where model_type can be "elife" or "elife_constant" '
             'and model_config can be a version.'
    ),
    *DEFAULT_POSREC_ALGO_OPTION
)
class CorrectedAreas(strax.Plugin):
    """
    Plugin which applies light collection efficiency maps and electron
    life time to the data.

    Computes the cS1/cS2 for the main/alternative S1/S2 as well as the
    corrected life time.

    Note:
        Please be aware that for both, the main and alternative S1, the
        area is corrected according to the xy-position of the main S2.
        
        There are now 3 components of cS2s: cs2_top, cS2_bottom and cs2.
        cs2_top and cs2_bottom are corrected by the corresponding maps,
        and cs2 is the sum of the two.
    """
    __version__ = '0.1.3'

    depends_on = ['event_basics', 'event_positions']

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for peak_type, peak_name in zip(['', 'alt_'], ['main', 'alternate']):
            dtype += [(f'{peak_type}cs1', np.float32, f'Corrected area of {peak_name} S1 [PE]'),
                      (f'{peak_type}cs2_wo_elifecorr', np.float32,
                       f'Corrected area of {peak_name} S2 before elife correction (s2 xy correction only) [PE]'),
                      (f'{peak_type}cs2_area_fraction_top', np.float32,
                       f'Fraction of area seen by the top PMT array for corrected {peak_name} S2'),
                      (f'{peak_type}cs2_bottom', np.float32,
                       f'Corrected area of {peak_name} S2 in the bottom PMT array [PE]'),
                      (f'{peak_type}cs2', np.float32, f'Corrected area of {peak_name} S2 [PE]'),]

        return dtype

    def setup(self):
        self.elife = get_correction_from_cmt(self.run_id, self.config['elife_conf'])

        self.s1_map = InterpolatingMap(
            get_cmt_resource(self.run_id,
                             tuple(['suffix',
                                    self.config['default_reconstruction_algorithm'],
                                    *strax.to_str_tuple(self.config['s1_xyz_correction_map'])]),
                             fmt='text'))

        self.s2_map = InterpolatingMap(
            get_cmt_resource(self.run_id,
                             tuple(['suffix',
                                    self.config['default_reconstruction_algorithm'],
                                    *strax.to_str_tuple(self.config['s2_xy_correction_map'])]),
                             fmt='json'))

    def compute(self, events):
        result = dict(
            time=events['time'],
            endtime=strax.endtime(events)
        )

        # S1 corrections depend on the actual corrected event position.
        # We use this also for the alternate S1; for e.g. Kr this is
        # fine as the S1 correction varies slowly.
        event_positions = np.vstack([events['x'], events['y'], events['z']]).T
        for peak_type in ["", "alt_"]:
            result[f"{peak_type}cs1"] = events[f'{peak_type}s1_area'] / self.s1_map(event_positions)

        # s2 corrections
        # S2 top and bottom are corrected separately, and cS2 total is the sum of the two
        # figure out the map name
        if len(self.s2_map.map_names) > 1:
            s2_top_map_name = "map_top"
            s2_bottom_map_name = "map_bottom"
        else:
            s2_top_map_name = "map"
            s2_bottom_map_name = "map"

        for peak_type in ["", "alt_"]:
            # S2(x,y) corrections use the observed S2 positions
            s2_positions = np.vstack([events[f'{peak_type}s2_x'], events[f'{peak_type}s2_y']]).T

            # corrected s2 with s2 xy map only, i.e. no elife correction
            # this is for s2-only events which don't have drift time info
            cs2_top_wo_elifecorr = (events[f'{peak_type}s2_area'] * events[f'{peak_type}s2_area_fraction_top'] / 
                                    self.s2_map(s2_positions, map_name=s2_top_map_name))
            cs2_bottom_wo_elifecorr = (events[f'{peak_type}s2_area'] *
                                       (1 - events[f'{peak_type}s2_area_fraction_top']) /
                                       self.s2_map(s2_positions, map_name=s2_bottom_map_name))
            result[f"{peak_type}cs2_wo_elifecorr"] = (cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr)

            # cs2aft doesn't need elife correction as it cancels
            result[f"{peak_type}cs2_area_fraction_top"] = cs2_top_wo_elifecorr / result[f"{peak_type}cs2_wo_elifecorr"]

            # For electron lifetime corrections to the S2s,
            # use drift time computed using the main S1.
            el_string = peak_type + "s2_interaction_" if peak_type == "alt_" else peak_type
            elife_correction = np.exp(events[f'{el_string}drift_time'] / self.elife)

            cs2_top = cs2_top_wo_elifecorr * elife_correction
            result[f"{peak_type}cs2_bottom"] = cs2_bottom_wo_elifecorr * elife_correction
            result[f"{peak_type}cs2"] = cs2_top + result[f"{peak_type}cs2_bottom"]

        return result


@export
@strax.takes_config(
    strax.Option(
        'g1', infer_type=False,
        help="S1 gain in PE / photons produced",
        default_by_run=[(0, 0.1442),
                        (first_sr1_run, 0.1426)]),
    strax.Option(
        'g2', infer_type=False,
        help="S2 gain in PE / electrons produced",
        default_by_run=[(0, 11.52/(1 - 0.63)),
                        (first_sr1_run, 11.55/(1 - 0.63))]),
    strax.Option(
        'lxe_w', infer_type=False,
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


@export
class EventShadow(strax.Plugin):
    """
    This plugin can calculate shadow at event level.
    It depends on peak-level shadow.
    The event-level shadow is its first S2 peak's shadow.
    If no S2 peaks, the event shadow will be nan.
    It also gives the position infomation of the previous S2s
    and main peaks' shadow.
    """
    __version__ = '0.0.8'
    depends_on = ('event_basics', 'peak_basics', 'peak_shadow')
    provides = 'event_shadow'
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = [('s1_shadow', np.float32, 'main s1 shadow [PE/ns]'),
                 ('s2_shadow', np.float32, 'main s2 shadow [PE/ns]'),
                 ('shadow', np.float32, 'shadow of event [PE/ns]'),
                 ('pre_s2_area', np.float32, 'previous s2 area [PE]'),
                 ('shadow_dt', np.int64, 'time difference to the previous s2 [ns]'),
                 ('shadow_index', np.int32, 'max shadow peak index in event'),
                 ('pre_s2_x', np.float32, 'x of previous s2 peak causing shadow [cm]'),
                 ('pre_s2_y', np.float32, 'y of previous s2 peak causing shadow [cm]'),
                 ('shadow_distance', np.float32, 'distance to the s2 peak with max shadow [cm]')]
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        res = np.zeros(len(events), self.dtype)

        res['shadow_index'] = -1
        res['pre_s2_x'] = np.nan
        res['pre_s2_y'] = np.nan

        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            if event['s1_index'] >= 0:
                res['s1_shadow'][event_i] = sp['shadow'][event['s1_index']]
            if event['s2_index'] >= 0:
                res['s2_shadow'][event_i] = sp['shadow'][event['s2_index']]
            if (sp['type'] == 2).sum() > 0:
                # Define event shadow as the first S2 peak shadow
                first_s2_index = np.argwhere(sp['type'] == 2)[0]
                res['shadow_index'][event_i] = first_s2_index
                res['shadow'][event_i] = sp['shadow'][first_s2_index]
                res['pre_s2_area'][event_i] = sp['pre_s2_area'][first_s2_index]
                res['shadow_dt'][event_i] = sp['shadow_dt'][first_s2_index]
                res['pre_s2_x'][event_i] = sp['pre_s2_x'][first_s2_index]
                res['pre_s2_y'][event_i] = sp['pre_s2_y'][first_s2_index]
        res['shadow_distance'] = ((res['pre_s2_x'] - events['s2_x'])**2 +
                                  (res['pre_s2_y'] - events['s2_y'])**2
                                  )**0.5
        res['time'] = events['time']
        res['endtime'] = strax.endtime(events)
        return res
