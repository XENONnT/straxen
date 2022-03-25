from re import X
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

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    dtype = [
        ('event_number', np.int64, 'Event number in this dataset'),
        ('time', np.int64, 'Event start time in ns since the unix epoch'),
        ('endtime', np.int64, 'Event end time in ns since the unix epoch')]

    events_seen = 0

    def setup(self):
        self.drift_time_max = int(self.config['max_drift_length'] / self.electron_drift_velocity)
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
        name='force_alt_s2_in_max_drift_time', default=True, infer_type=False,
        help="Make sure alt_s2 is in max drift time starting from main S1"),
    strax.Option(
        name='event_s1_min_coincidence',
        default=2, infer_type=False,
        help="Event level S1 min coincidence. Should be >= s1_min_coincidence "
             "in the peaklet classification"),
    strax.Option(
        name='max_drift_length',
        default=straxen.tpc_z, infer_type=False,
        help='Total length of the TPC from the bottom of gate to the '
             'top of cathode wires [cm]',),
)
class EventBasics(strax.Plugin):
    """
    Computes the basic properties of the main/alternative S1/S2 within
    an event.

    The main S1 and alternative S1 are given by the largest two S1-Peaks
    within the event.
    The main S2 is given by the largest S2-Peak within the event, while
    alternative S2 is selected as the largest S2 other than main S2
    in the time window [main S1 time, main S1 time + max drift time].
    """
    __version__ = '1.3.1'

    depends_on = ('events',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')
    provides = 'event_basics'
    data_kind = 'events'
    loop_over = 'events'

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

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
            ('n_hits',            np.int16,   'count of hits contributing at least one sample to the peak'),
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

    def setup(self):
        self.drift_time_max = int(self.config['max_drift_length'] / self.electron_drift_velocity)

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
        # number_of_peaks=0 selects all available s2 and sort by area
        largest_s2s, s2_idx = self.get_largest_sx_peaks(peaks, s_i=2, number_of_peaks=0)

        if not self.config['allow_posts2_s1s'] and len(largest_s2s):
            s1_latest_time = largest_s2s[0]['time']
        else:
            s1_latest_time = np.inf

        largest_s1s, s1_idx = self.get_largest_sx_peaks(
            peaks,
            s_i=1,
            s1_before_time=s1_latest_time,
            s1_min_coincidence=self.config['event_s1_min_coincidence'])

        if self.config['force_alt_s2_in_max_drift_time']:
            s2_idx, largest_s2s = self.find_main_alt_s2(largest_s1s,
                                                        s2_idx,
                                                        largest_s2s,
                                                        self.drift_time_max,
                                                        )
        else:
            # Select only the largest two S2s
            largest_s2s, s2_idx = largest_s2s[0:2], s2_idx[0:2]

        if self.config['force_main_before_alt']:
            s2_order = np.argsort(largest_s2s['time'])
            largest_s2s = largest_s2s[s2_order]
            s2_idx = s2_idx[s2_order]

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
    def find_main_alt_s2(largest_s1s, s2_idx, largest_s2s, drift_time_max):
        """Require alt_s2 happens between main S1 and maximum drift time"""
        if len(largest_s1s) > 0 and len(largest_s2s) > 1:
            # If there is a valid s1-s2 pair and has a second s2, then check alt s2 validity
            s2_after_s1 = largest_s2s['center_time'] > largest_s1s[0]['center_time']
            s2_before_max_drift_time = (largest_s2s['center_time']
                                        - largest_s1s[0]['center_time']) < 1.01 * drift_time_max
            mask = s2_after_s1 & s2_before_max_drift_time
            # The selection avoids main_S2
            mask[0] = True
            # Take main and the largest valid alt_S2
            s2_idx, largest_s2s = s2_idx[mask][:2], largest_s2s[mask][:2]
        return s2_idx, largest_s2s

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
    __version__ = '0.2.1'

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
        default='cmt://avg_se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Nominal single electron (SE) gain in PE / electron extracted. '
             'Data will be corrected to this value')

    # se gain for this run, allowing for using CMT. default to online
    se_gain = straxen.URLConfig(
        default='cmt://se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Actual SE gain for a given run (allows for time dependence)')

    # relative extraction efficiency which can change with time and modeled by CMT.
    rel_extraction_eff = straxen.URLConfig(
        default='cmt://rel_extraction_eff?version=ONLINE&run_id=plugin.run_id',
        help='Relative extraction efficiency for this run (allows for time dependence)')

    # relative light yield
    # defaults to no correction
    rel_light_yield = straxen.URLConfig(
        default='cmt://relative_light_yield?version=ONLINE&run_id=plugin.run_id',
        help='Relative light yield (allows for time dependence)'
    )

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for peak_type, peak_name in zip(['', 'alt_'], ['main', 'alternate']):
            dtype += [(f'{peak_type}cs1', np.float32, f'Corrected area of {peak_name} S1 [PE]'),
                      (f'{peak_type}cs1_wo_timecorr', np.float32,
                       f'Corrected area of {peak_name} S1 [PE] before time-dep LY correction'),
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
            result[f"{peak_type}cs1_wo_timecorr"] = events[f'{peak_type}s1_area'] / self.s1_xyz_map(event_positions)
            result[f"{peak_type}cs1"] = result[f"{peak_type}cs1_wo_timecorr"] / self.rel_light_yield

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
class EventShadow(strax.Plugin):
    """
    This plugin can calculate shadow for main S1 and main S2 in events.
    It also gives the position information of the previous peaks.
    References:
        * v0.1.4 reference: xenon:xenonnt:ac:prediction:shadow_ambience
    """
    __version__ = '0.1.4'
    depends_on = ('event_basics', 'peak_basics', 'peak_shadow')
    provides = 'event_shadow'

    def infer_dtype(self):
        dtype = []
        for main_peak, main_peak_desc in zip(['s1_', 's2_'], ['main S1', 'main S2']):
            # previous S1 can only cast time shadow, previous S2 can cast both time & position shadow
            for key in ['s1_time_shadow', 's2_time_shadow', 's2_position_shadow']:
                type_str, tp_desc, _ = key.split('_')
                dtype.append(((f'largest {tp_desc} shadow casting from previous {type_str} to {main_peak_desc} [PE/ns]', 
                               f'{main_peak}shadow_{key}'), np.float32))
                dtype.append(((f'time difference from the previous {type_str} casting largest {tp_desc} shadow to {main_peak_desc} [ns]', 
                               f'{main_peak}dt_{key}'), np.int64))
                # Only previous S2 peaks have (x,y)
                if 's2' in key:
                    dtype.append(((f'x of previous s2 peak casting largest {tp_desc} shadow on {main_peak_desc} [cm]', 
                                   f'{main_peak}x_{key}'), np.float32))
                    dtype.append(((f'y of previous s2 peak casting largest {tp_desc} shadow on {main_peak_desc} [cm]', 
                                   f'{main_peak}y_{key}'), np.float32))
                # Only time shadow gives the nearest large peak
                if 'time' in key:
                    dtype.append(((f'time difference from the nearest previous large {type_str} to {main_peak_desc} [ns]', 
                                   f'{main_peak}nearest_dt_{type_str}'), np.int64))
            # Also record the PDF of HalfCauchy when calculating S2 position shadow
            dtype.append(((f'PDF describing correlation between previous s2 and {main_peak_desc}', 
                           f'{main_peak}pdf_s2_position_shadow'), np.float32))
        dtype += strax.time_fields
        return dtype

    @staticmethod
    def set_nan_defaults(result):
        """
        When constructing the dtype, take extra care to set values to
        np.Nan / -1 (for ints) as 0 might have a meaning
        """
        for field in result.dtype.names:
            if np.issubdtype(result.dtype[field], np.integer):
                result[field][:] = -1
            else:
                result[field][:] = np.nan

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.dtype)

        self.set_nan_defaults(result)

        # 1. Assign peaks features to main S1 and main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            res_i = result[event_i]
            # Fetch the features of main S1 and main S2
            for idx, main_peak in zip([event['s1_index'], event['s2_index']], ['s1_', 's2_']):
                if idx >= 0:
                    for key in ['s1_time_shadow', 's2_time_shadow', 's2_position_shadow']:
                        type_str = key.split('_')[0]
                        res_i[f'{main_peak}shadow_{key}'] = sp[f'shadow_{key}'][idx]
                        res_i[f'{main_peak}dt_{key}'] = sp[f'dt_{key}'][idx]
                        if 'time' in key:
                            res_i[f'{main_peak}nearest_dt_{type_str}'] = sp[f'nearest_dt_{type_str}'][idx]
                        if 's2' in key:
                            res_i[f'{main_peak}x_{key}'] = sp[f'x_{key}'][idx]
                            res_i[f'{main_peak}y_{key}'] = sp[f'y_{key}'][idx]
                    # Record the PDF of HalfCauchy
                    res_i[f'{main_peak}pdf_s2_position_shadow'] = sp['pdf_s2_position_shadow'][idx]

        # 2. Set time and endtime for events
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        return result


@export
class EventAmbience(strax.Plugin):
    """
    Save Ambience of the main S1 and main S2 in the event.
    References:
        * v0.0.4 reference: xenon:xenonnt:ac:prediction:shadow_ambience
    """
    __version__ = '0.0.4'
    depends_on = ('event_basics', 'peak_basics', 'peak_ambience')
    provides = 'event_ambience'

    @property
    def origin_dtype(self):
        return ['lh_before', 's0_before', 's1_before', 's2_before', 's2_near']

    def infer_dtype(self):
        dtype = []
        for ambience in self.origin_dtype:
            dtype.append(((f"Number of  {' '.join(ambience.split('_'))} main S1", 
                           f's1_n_{ambience}'), np.int16))
            dtype.append(((f"Number of  {' '.join(ambience.split('_'))} main S2", 
                           f's2_n_{ambience}'), np.int16))
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)

        # 1. Initialization, ambience is set to be the lowest possible value
        result = np.zeros(len(events), self.dtype)

        # 2. Assign peaks features to main S1, main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            for idx, main_peak in zip([event['s1_index'], event['s2_index']], ['s1_', 's2_']):
                if idx >= 0:
                    for ambience in self.origin_dtype:
                        result[f'{main_peak}n_{ambience}'][event_i] = sp[f'n_{ambience}'][idx]

        # 3. Set time and endtime for events
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        return result
