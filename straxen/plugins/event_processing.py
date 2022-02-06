import strax
import numpy as np
import numba
import straxen
from .position_reconstruction import DEFAULT_POSREC_ALGO
from straxen.common import pax_file, get_resource, first_sr1_run
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

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO,
        help="default reconstruction algorithm that provides (x,y)"
    )

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

        self.electron_drift_velocity = get_correction_from_cmt(
            self.run_id, self.config['electron_drift_velocity'])
        self.electron_drift_time_gate = get_correction_from_cmt(
            self.run_id, self.config['electron_drift_time_gate'])
        
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
    __version__ = '0.2.0'

    depends_on = ['event_basics', 'event_positions']

    # Descriptor configs
    elife = straxen.URLConfig(
        default='cmt://elife?version=ONLINE&run_id=plugin.run_id',
        help='electron lifetime in [ns]')

    # default posrec, used to determine which LCE map to use
    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO,
        help="default reconstruction algorithm that provides (x,y)"
    )
    s1_xyz_map = straxen.URLConfig(
        default='itp_map://resource://cmt://format://'
                's1_xyz_map_{algo}?version=ONLINE&run_id=plugin.run_id'
                '&fmt=json&algo=plugin.default_reconstruction_algorithm',
        cache=True)
    s2_xy_map = straxen.URLConfig(
        default='itp_map://resource://cmt://format://'
                's2_xy_map_{algo}?version=ONLINE&run_id=plugin.run_id'
                '&fmt=json&algo=plugin.default_reconstruction_algorithm',
        cache=True)

    # average SE gain for a given time period. default to the value of this run in ONLINE model
    # thus, by default, there will be no time-dependent correction according to se gain
    avg_se_gain = straxen.URLConfig(
        default='cmt://se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Nominal single electron (SE) gain in PE / electron extracted. '
             'Data will be corrected to this value')

    # se gain for this run, allowing for using CMT. default to online
    se_gain = straxen.URLConfig(
        default='cmt://se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Actual SE gain for a given run (allows for time dependence)')

    # relative extraction efficiency which can change with time and modeled by CMT.
    # defaults to no correction
    rel_extraction_eff = straxen.URLConfig(
        default=1.0,
        help='Relative extraction efficiency for this run (allows for time dependence)')

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for peak_type, peak_name in zip(['', 'alt_'], ['main', 'alternate']):
            dtype += [(f'{peak_type}cs1', np.float32, f'Corrected area of {peak_name} S1 [PE]'),
                      (f'{peak_type}cs2_wo_elifecorr', np.float32,
                       f'Corrected area of {peak_name} S2 before elife correction '
                       f'(s2 xy correction + SEG/EE correction applied) [PE]'),
                      (f'{peak_type}cs2_wo_timecorr', np.float32,
                       f'Corrected area of {peak_name} S2 before SEG/EE and elife corrections'
                       f'(s2 xy correction applied) [PE]'),
                      (f'{peak_type}cs2_area_fraction_top', np.float32,
                       f'Fraction of area seen by the top PMT array for corrected {peak_name} S2'),
                      (f'{peak_type}cs2_bottom', np.float32,
                       f'Corrected area of {peak_name} S2 in the bottom PMT array [PE]'),
                      (f'{peak_type}cs2', np.float32, f'Corrected area of {peak_name} S2 [PE]'), ]
        return dtype

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
            result[f"{peak_type}cs1"] = events[f'{peak_type}s1_area'] / self.s1_xyz_map(event_positions)

        # s2 corrections
        # S2 top and bottom are corrected separately, and cS2 total is the sum of the two
        # figure out the map name
        if len(self.s2_xy_map.map_names) > 1:
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
            cs2_top_xycorr = (events[f'{peak_type}s2_area'] * events[f'{peak_type}s2_area_fraction_top'] /
                                    self.s2_xy_map(s2_positions, map_name=s2_top_map_name))
            cs2_bottom_xycorr = (events[f'{peak_type}s2_area'] *
                                       (1 - events[f'{peak_type}s2_area_fraction_top']) /
                                       self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name))

            # Correct for SEgain and extraction efficiency
            seg_ee_corr = (self.se_gain / self.avg_se_gain) * self.rel_extraction_eff
            cs2_top_wo_elifecorr = cs2_top_xycorr / seg_ee_corr
            cs2_bottom_wo_elifecorr = cs2_bottom_xycorr / seg_ee_corr
            result[f"{peak_type}cs2_wo_elifecorr"] = cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr

            # cs2aft doesn't need elife/time corrections as they cancel
            result[f"{peak_type}cs2_area_fraction_top"] = cs2_top_wo_elifecorr / result[f"{peak_type}cs2_wo_elifecorr"]


            # For electron lifetime corrections to the S2s,
            # use drift time computed using the main S1.
            el_string = peak_type + "s2_interaction_" if peak_type == "alt_" else peak_type
            elife_correction = np.exp(events[f'{el_string}drift_time'] / self.elife)
            result[f"{peak_type}cs2_wo_timecorr"] = (cs2_top_xycorr + cs2_bottom_xycorr) * elife_correction
            result[f"{peak_type}cs2"] = result[f"{peak_type}cs2_wo_elifecorr"] * elife_correction
            result[f"{peak_type}cs2_bottom"] = cs2_bottom_wo_elifecorr * elife_correction

        return result


@export
class EnergyEstimates(strax.Plugin):
    """
    Plugin which converts cS1 and cS2 into energies (from PE to KeVee).
    """
    __version__ = '0.1.1'
    depends_on = ['corrected_areas']
    dtype = [
        ('e_light', np.float32, 'Energy in light signal [keVee]'),
        ('e_charge', np.float32, 'Energy in charge signal [keVee]'),
        ('e_ces', np.float32, 'Energy estimate [keVee]')
    ] + strax.time_fields
    save_when = strax.SaveWhen.TARGET

    # config options don't double cache things from the resource cache!
    g1 = straxen.URLConfig(
        default='bodega://g1?bodega_version=v2',
        help="S1 gain in PE / photons produced",
    )
    g2 = straxen.URLConfig(
        default='bodega://g2?bodega_version=v2',
        help="S2 gain in PE / electrons produced",
    )
    lxe_w = straxen.URLConfig(
        default=13.7e-3,
        help="LXe work function in quanta/keV"
    )

    def compute(self, events):
        el = self.cs1_to_e(events['cs1'])
        ec = self.cs2_to_e(events['cs2'])
        return dict(e_light=el,
                    e_charge=ec,
                    e_ces=el + ec,
                    time=events['time'],
                    endtime=strax.endtime(events))

    def cs1_to_e(self, x):
        return self.lxe_w * x / self.g1

    def cs2_to_e(self, x):
        return self.lxe_w * x / self.g2


@export
@strax.takes_config(
    strax.Option(name='pre_s1_area_threshold', default=1e3,
                 help='Only take S1s larger than this into account '
                      'when calculating PeakShadow [PE]'),
    strax.Option(name='pre_s2_area_threshold', default=1e4,
                 help='Only take S2s larger than this into account '
                      'when calculating PeakShadow [PE]'),
    strax.Option(name='deltatime_exponent', default=-1.0,
                 help='The exponent of delta t when calculating shadow'),
    strax.Option(name='time_window_backward', default=int(1e9),
                 help='Search for peaks casting shadow in this time window [ns]')
)
class EventShadow(strax.Plugin):
    """
    This plugin can calculate shadow at event level.
    It depends on peak-level shadow.
    The event-level shadow is its first peak peak's shadow.
    It also gives the position infomation of the previous peaks
    and main peaks' shadow.
    References:
        * v0.1.0 reference: xenon:xenonnt:ac:prediction:shadow_ambience
    """
    __version__ = '0.1.0'
    depends_on = ('event_basics', 'peak_basics', 'peak_shadow')
    provides = 'event_shadow'
    data_kind = 'events'
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = []
        for sa, r in zip(['', 'alt_'], ['1st', '2nd']):
            for s in ['s1', 's2']:
                for p, si in zip(['s1', 's2', 's2_re'], ['s1', 's2', 'reordered s2']):
                    dtype.append((('main ' + s + ' shadow from ' + si + ' casting ' + r + ' largest shadow [PE/ns]', s + '_' + sa + 'shadow_' + p), np.float32))
                    dtype.append((('previous ' + si + ' area casting ' + r + ' largest shadow on ' + s + ' [PE]', s + '_' + sa + 'pre_area_' + p), np.float32))
                    dtype.append((('time difference from ' + s + ' to the previous ' + si + ' casting ' + r + ' largest shadow [ns]', s + '_' + sa + 'shadow_dt_' + p), np.int64))
            for p, si in zip(['s1', 's2', 's2_re'], ['s1', 's2', 'reordered s2']):
                dtype.append((('event shadow from ' + si + ' casting ' + r + ' largest shadow [PE/ns]', sa + 'shadow_' + p), np.float32))
                dtype.append((('previous ' + si + ' area casting ' + r + ' largest shadow  on the event [PE]', sa + 'pre_area_' + p), np.float32))
                dtype.append((('time difference from the event to the previous ' + si + ' casting ' + r + ' largest shadow [ns]', sa + 'shadow_dt_' + p), np.int64))
            for x in ['x', 'y']:
                dtype.append(((x + ' of previous s2 peak casting ' + r + ' largest shadow [cm]', sa + 'pre_' + x + '_s2'), np.float32))
                dtype.append(((x + ' of previous reordered s2 peak casting ' + r + ' largest shadow [cm]', sa + 'pre_' + x + '_s2_re'), np.float32))
            dtype.append((('distance to the s2 peak with ' + r + ' largest shadow [cm]', sa + 'shadow_distance'), np.float32))
            dtype.append((('distance to the reordered s2 peak with ' + r + ' largest shadow [cm]', sa + 'shadow_distance_re'), np.float32))
            dtype.append((('previous ' + r + ' largest s2 shadow with position correlation PDF [PE/ns]', sa + 'shadow_s2_corr'), np.float32))
            dtype.append((('previous ' + r + ' largest s2 shadow position correlation PDF', sa + 'shadow_s2_prob'), np.float32))
        dtype.append((('index of the peak defining the event shadow', 'shadow_index'), np.int32))
        for s in ['s1', 's2']:
            for p, si in zip(['s1', 's2', 's2_re'], ['s1', 's2', 'reordered s2']):
                dtype.append((('Nearest ' + si + ' delta t to ' + s, s + '_near_dt_' + p), np.int64))
                dtype.append((('Sum of ' + si + ' shadow in time window casting on ' + s, s + '_shadow_sum_' + p), np.float32))
        for p, si in zip(['s1', 's2', 's2_re'], ['s1', 's2', 'reordered s2']):
            dtype.append((('Nearest ' + si + ' delta t to event', 'near_dt_' + p), np.int64))
            dtype.append((('Sum of ' + si + ' shadow in time window casting on event', 'shadow_sum_' + p), np.float32))
        dtype += strax.time_fields
        return dtype

    def setup(self):
        self.time_window_backward = self.config['time_window_backward']
        self.threshold = dict(s1=self.config['pre_s1_area_threshold'], s2=self.config['pre_s2_area_threshold'], s2_re=self.config['pre_s2_area_threshold'])
        self.exponent = self.config['deltatime_exponent']

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        res = np.zeros(len(events), self.dtype)

        res['shadow_index'] = -1

        # Set default values
        for sa in ['', 'alt_']:
            for key in ['s1_', 's2_', '']:
                for s, ty in zip(['s1', 's2', 's2_re'], ['s1', 's2', 's2_re']):
                    res[key + sa + 'pre_area_' + s] = self.threshold[ty]
                    res[key + sa + 'shadow_dt_' + s] = self.time_window_backward
                    res[key + sa + 'shadow_' + s] = res[key + sa + 'pre_area_' + s] * res[key + sa + 'shadow_dt_' + s] ** self.exponent
            for re in ['', '_re']:
                for x in ['x', 'y']:
                    res[sa + 'pre_' + x + '_s2' + re] = np.nan
                res[sa + 'shadow_distance' + re] = np.nan
            res[sa + 'shadow_s2_prob'] = np.nan
            res[sa + 'shadow_s2_corr'] = np.nan
        for key in ['s1_', 's2_', '']:
            for s, ty in zip(['s1', 's2', 's2_re'], ['s1', 's2', 's2_re']):
                res[key + 'near_dt_' + s] = self.time_window_backward
                res[key + 'shadow_sum_' + s] = self.threshold[ty] * self.time_window_backward ** self.exponent

        # Assign peaks features to events
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            indices = [event['s1_index'], event['s2_index'], np.argwhere(sp['type'] == 2)[0] if (sp['type'] == 2).sum() > 0 else -1]
            for idx, key in zip(indices, ['s1_', 's2_', '']):
                if idx >= 0:
                    for s in ['s1', 's2', 's2_re']:
                        for sa in ['', 'alt_']:
                            res[key + sa + 'pre_area_' + s][event_i] = sp[sa + 'pre_area_' + s][idx]
                            res[key + sa + 'shadow_' + s][event_i] = sp[sa + 'shadow_' + s][idx]
                            res[key + sa + 'shadow_dt_' + s][event_i] = sp[sa + 'shadow_dt_' + s][idx]
                        res[key + 'near_dt_' + s][event_i] = sp['near_dt_' + s][idx]
                        res[key + 'shadow_sum_' + s][event_i] = sp['shadow_sum_' + s][idx]
            if (sp['type'] == 2).sum() > 0:
                res['shadow_index'][event_i] = indices[-1]
                for sa in ['', 'alt_']:
                    res[sa + 'shadow_s2_corr'][event_i] = sp[sa + 'shadow_s2_corr'][indices[-1]]
                    res[sa + 'shadow_s2_prob'][event_i] = sp[sa + 'shadow_s2_prob'][indices[-1]]
                    for x in ['x', 'y']:
                        for re in ['', '_re']:
                            res[sa + 'pre_' + x + '_s2' + re][event_i] = sp[sa + 'pre_' + x + '_s2' + re][indices[-1]]
        for sa in ['', 'alt_']:
            for re in ['', '_re']:
                res[sa + 'shadow_distance' + re] = ((res[sa + 'pre_x_s2' + re] - events['s2_x'])**2 + 
                                                   (res[sa + 'pre_y_s2' + re] - events['s2_y'])**2) ** 0.5
        res['time'] = events['time']
        res['endtime'] = strax.endtime(events)
        return res


@export
class EventAmbience(strax.Plugin):
    """
    This plugin can calculate ambience at event level.
    Save Ambience of the main S1 peak, main S2 peaak and the first S2 peak in the event.
    References:
        * v0.0.1 reference: xenon:xenonnt:ac:prediction:shadow_ambience
    """
    __version__ = '0.0.1'
    depends_on = ('event_basics', 'peaks', 'peak_basics', 'peak_ambience')
    provides = 'event_ambience'
    save_when = strax.SaveWhen.EXPLICIT

    @property
    def origindtype(self):
        return ['lonehit_before', 's0_before', 's1_before', 's2_before'] + ['s2_near']

    def infer_dtype(self):
        dtype = []
        for si in self.origindtype:
            for s in ['', '_s1', '_s2']:
                if '' == s:
                    sig = 'event'
                else:
                    sig = s.split('_')[1]
                dtype.append((('Number of ' + ' '.join(si.split('_')) + ' an ' + sig, 'n_' + si + s), np.int16))
        dtype.append(('s1_n_hits', np.int32)) 
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        res = self.compute_s1_s2(events, peaks)
        res['s1_n_hits'] = self.s1_hits(events, peaks)
        return res

    def compute_s1_s2(self, events, peaks):
        split_peaks = strax.touching_windows(peaks, events)
        res = np.zeros(len(events), self.dtype)

        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            indices = [event['s1_index'], event['s2_index'], np.argwhere(peaks['type'][sp[0]:sp[-1]] == 2)[0] if (peaks['type'][sp[0]:sp[-1]] == 2).sum() > 0 else -1]
            for idx, key in zip(indices, ['_s1', '_s2', '']):
                if idx >= 0:
                    for si in self.origindtype:
                        res['n_' + si + key][event_i] = peaks['n_' + si][sp[0]:sp[-1]][idx]
        res['time'] = events['time']
        res['endtime'] = strax.endtime(events)
        return res

    def s1_hits(self, events, peaks):
        res = np.full(len(events), -1).astype(np.int32)
        touching_windows = strax.touching_windows(peaks, events)
        for event_i, (event, indices) in enumerate(zip(events, touching_windows)):
            if event['s1_index'] != -1:
                res[event_i] = peaks['n_hits'][indices[0]:indices[1]][event['s1_index']]
        return res
