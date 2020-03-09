import strax

from straxen.common import pax_file, get_resource, get_elife, first_sr1_run
from straxen.itp_map import InterpolatingMap
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('n_top_pmts', default=127,
                 help="Number of top PMTs"))
class DoubleS1Scatter_PeakBasics(strax.Plugin):
    __version__ = "0.1.0"
    parallel = True
    depends_on = ('peaks',)
    provides = 'DoubleS1Scatter_peak_basics'
    dtype = [
        (('Start time of the peak (ns since unix epoch)',
          'time'), np.int64),
        (('End time of the peak (ns since unix epoch)',
          'endtime'), np.int64),
        (('Weighted center time of the peak (ns since unix epoch)',
          'center_time'), np.int64),
        (('Peak integral in PE',
            'area'), np.float32),
        (('Number of PMTs contributing to the peak',
            'n_channels'), np.int16),
        (('PMT number which contributes the most PE',
            'max_pmt'), np.int16),
        (('Area of signal in the largest-contributing PMT (PE)',
            'max_pmt_area'), np.int32),
        (('Width (in ns) of the central 50% area of the peak',
            'range_50p_area'), np.float32),
        (('Width (in ns) of the central 90% area of the peak',
            'range_90p_area'), np.float32),
        (('Fraction of area seen by the top array',
            'area_fraction_top'), np.float32),
        (('Length of the peak waveform in samples',
          'length'), np.int32),
        (('Time resolution of the peak waveform in ns',
          'dt'), np.int16),
        (('Time between 10% and 50% area quantiles [ns]',
          'rise_time'), np.float32),
        (('Hits within tight range of mean',
          'tight_coincidence'), np.int16),
        (('Classification of the peak(let)',
          'type'), np.int8),
        (('PMT numbers contributing to the peak',
          'contributing_channel'), (np.int16,(248)))
    ]

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), self.dtype)
        for q in 'time length dt area type'.split():
            r[q] = p[q]
        r['endtime'] = p['time'] + p['dt'] * p['length']
        r['n_channels'] = (p['area_per_channel'] > 0).sum(axis=1)
        r['range_50p_area'] = p['width'][:, 5]
        r['range_90p_area'] = p['width'][:, 9]
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)
        r['max_pmt_area'] = np.max(p['area_per_channel'], axis=1)
        r['tight_coincidence'] = p['tight_coincidence']

        r['center_time'] = p['time'] + self.compute_center_times(peaks)

        n_top = self.config['n_top_pmts']
        area_top = p['area_per_channel'][:, :n_top].sum(axis=1)
        # Negative-area peaks get 0 AFT - TODO why not NaN?
        m = p['area'] > 0
        r['area_fraction_top'][m] = area_top[m]/p['area'][m]
        r['rise_time'] = -p['area_decile_from_midpoint'][:,1]
        r['contributing_channel']= (p['area_per_channel']>0)*1
        return r

    @staticmethod
    @numba.njit(cache=True, nogil=True)
    def compute_center_times(peaks):
        result = np.zeros(len(peaks), dtype=np.int32)
        for p_i, p in enumerate(peaks):
            t = 0
            for t_i, weight in enumerate(p['data']):
                t += t_i * p['dt'] * weight
            result[p_i] = t / p['area']
        return result


    
@export
class DoubleS1Scatter_EventBasics(strax.LoopPlugin):
    __version__ = "0.1.0"
    depends_on = ('events',
                  'DoubleS1Scatter_peak_basics',
                  'peak_positions',
                  'peak_proximity')
    provides = 'DoubleS1Scatter_event_basics'

    def infer_dtype(self):
        dtype = [(('Number of peaks in the event',
                   'n_peaks'), np.int32),
                 (('Drift time between S1_a and S2_a in ns',
                   'drift_time'), np.int64)]
        for i in [1, 2]:
            dtype += [((f'Main S{i} peak index',
                        f's{i}_index'), np.int32),
                      ((f'Main S{i} area fraction top',
                        f's{i}_area_fraction_top'), np.float32),
                      ((f'Main S{i} width (ns, 50% area)',
                        f's{i}_range_50p_area'), np.float32),
                      ((f'Main S{i} weighted center time since unix epoch [ns]',
                        f's{i}_a_center_time'), np.int64),
                      ((f'Alternate S{i} weighted center time since unix epoch [ns]',
                        f's{i}_b_center_time'), np.int64),
                      ((f'Main S{i} number of competing peaks',
                        f's{i}_n_competing'), np.int32)]

        dtype += [((f'S1_a area (PE), uncorrected',
                    f's1_a_area'), np.float32),
                  ((f'S2_a area (PE), uncorrected',
                    f's2_a_area'), np.float32)]

        dtype += [(f's1_b_area',np.float32,
                   f'Largest other S1 area (PE) in event, uncorrected',),
                  (f's2_b_area',np.float32,
                   f'Largest other S2 area (PE) in event, uncorrected',)]

        dtype += [(f'x_s2_a', np.float32,
                   f'S2_a reconstructed X position (cm), uncorrected',),
                  (f'y_s2_a', np.float32,
                   f'S2_a reconstructed Y position (cm), uncorrected',),
                  (f'x_s2_b', np.float32,
                   f'S2_b reconstructed X position (cm), uncorrected',),
                  (f'y_s2_b', np.float32,
                   f'S2_b reconstructed Y position (cm), uncorrected',)]

        dtype += [(f'ds_s1_dt', np.int64,
                   f'Delay time between s1_a_center_time and s1_b_center_time',),
                  (f'ds_s2_dt', np.int64,
                   f'Delay time between s2_a_center_time and s2_b_center_time',),
                  (f'ds_s1_b_n_distinct_channels', np.int16,
                   f'number of PMTs contributing to s1_b distinct from the PMTs that contributed to s1_a',),
                 (f's1_a_n_contributing', np.int16,
                   f'number of PMTs contributing to s1_a',),
                 (f's1_b_n_contributing', np.int16,
                   f'number of PMTs contributing to s1_b ',)]
        dtype += strax.time_fields

        return dtype



    def compute_loop(self, event, peaks):
        result = dict(n_peaks=len(peaks),
                      time=event['time'],
                      endtime=strax.endtime(event))
        if not len(peaks):
            return result

        main_s = dict()
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

            # the S1_0 peak is the main S1
            if s_i == 1:
                s1_0_center_time = ss['center_time'][main_i]

            # the S2_0 peak is the main S2
            if s_i == 2:
                s2_0_center_time = ss['center_time'][main_i]
            #If only one S2 peak in the event -> ds_s2_dt = 0 ....
            if s_i == 2 and ss['n_competing'][main_i]==0:
                result[f'ds_s2_dt'] = 0
                result['s2_b_area'] = 0
                result['x_s2_b'] = 0
                result['y_s2_b'] = 0



            #Find largest other S2 signals
            if s_i == 2 and ss['n_competing'][main_i]>0 and len(ss['area'])>1:
                s2_second_i = np.argsort(ss['area'])[-2]
                s2_1_center_time = ss['center_time'][s2_second_i]

                if s2_1_center_time > s2_0_center_time:
                    s2_a_center_time = s2_0_center_time #S2_a = main
                    s2_b_center_time =  s2_1_center_time #S2_b = second
                    s2_a_area = ss['area'][main_i]
                    s2_b_area = ss['area'][s2_second_i]
                    s2_b_x = ss['x'][s2_second_i]
                    s2_b_y = ss['y'][s2_second_i]

                    #In this case main_i correspond to the first peak in time
                    #so we don't have to switch main_i and s1_second_i
                if s2_0_center_time >= s2_1_center_time:
                    s2_a_center_time = s2_1_center_time #S1_a = second
                    s2_b_center_time = s2_0_center_time #S1_b = main
                    s2_a_area = ss['area'][s2_second_i]
                    s2_b_area = ss['area'][main_i]
                    s2_b_x = ss['x'][main_i]
                    s2_b_y = ss['y'][main_i]
                    #In this case we have to switch main_i and s1_second_i
                    # to compute 'area_fraction_top','range_50p_area', 'n_competing' for S2_a
                    temp=main_i
                    main_i=s2_second_i
                    s2_second_i=temp

                result['s2_a_center_time'] = s2_a_center_time
                result['s2_b_center_time'] = s2_b_center_time
                result[f'ds_s2_dt'] = s2_b_center_time - s2_a_center_time
                result['s2_b_area'] = s2_b_area
                result['x_s2_b'] = s2_b_x
                result['y_s2_b'] = s2_b_y


            #If only one S1 peak in the event -> ds_s1_dt = 0 .....
            if s_i == 1 and ss['n_competing'][main_i]==0:
                result[f'ds_s1_dt'] = 0
                result['s1_b_area'] = 0

            #Else we take the two S1s peaks with the largest area and we order them by time
            if s_i == 1 and ss['n_competing'][main_i]>0 and len(ss['area'])>1:
                s1_second_i = np.argsort(ss['area'])[-2]
                s1_1_center_time = ss['center_time'][s1_second_i]

                if s1_1_center_time > s1_0_center_time:
                    s1_a_center_time = s1_0_center_time #S1_a = main
                    s1_b_center_time =  s1_1_center_time #S1_b = second
                    peaks1 = ss['contributing_channel'][main_i] #s1_a
                    peaks2 = ss['contributing_channel'][s1_second_i] #s1_b
                    s1_a_area = ss['area'][main_i]
                    s1_b_area = ss['area'][s1_second_i]

                    #In this case main_i correspond to the first peak in time
                    #so we don't have to switch main_i and s1_second_i
                if s1_0_center_time >= s1_1_center_time:
                    s1_a_center_time = s1_1_center_time #S1_a = second
                    s1_b_center_time = s1_0_center_time #S1_b = main
                    peaks1 = ss['contributing_channel'][s1_second_i] #s1_a
                    peaks2 = ss['contributing_channel'][main_i] #s1_b
                    s1_a_area = ss['area'][s1_second_i]
                    s1_b_area = ss['area'][main_i]
                    #In this case we have to switch main_i and s1_second_i
                    # to compute 'area_fraction_top','range_50p_area', 'n_competing' for S1_a
                    temp=main_i
                    main_i=s1_second_i
                    s1_second_i=temp

                result['s1_a_center_time'] = s1_a_center_time
                result['s1_b_center_time'] = s1_b_center_time
                result[f'ds_s1_dt'] = s1_b_center_time - s1_a_center_time
                result['s1_b_area'] = s1_b_area

                #compute variables related to the numbers of PMTs contributing to S1_a and S1_b,
                # and PMTs contributing to S1_b which are not contributing to S1_a
                s1_a_peaks = np.nonzero(peaks1)
                s1_b_peaks = np.nonzero(peaks2)

                result[f's1_a_n_contributing'] = len(s1_a_peaks[0])
                result[f's1_b_n_contributing'] = len(s1_b_peaks[0])

                ds_s1_b_n_distinct_channels=0
                for channel in range(len(s1_b_peaks[0])):
                    if s1_b_peaks[0][channel] not in s1_a_peaks[0]:
                        ds_s1_b_n_distinct_channels += 1
                result[f'ds_s1_b_n_distinct_channels'] = ds_s1_b_n_distinct_channels

            result[f's{s_i}_index'] = s_indices[main_i]
            s = main_s[s_i] = ss[main_i]

            for prop in ['area_fraction_top',
                         'range_50p_area', 'n_competing']:
                result[f's{s_i}_{prop}'] = s[prop]

            if s_i == 1:
                result['s1_a_area'] = s['area']
            if s_i == 2:
                result['s2_a_area'] = s['area']
                result['x_s2_a'] = s['x']
                result['y_s2_a'] = s['y']


        # Compute a drift time only if we have a valid S1-S2 pairs
        if len(main_s) == 2:
            result['drift_time'] = main_s[2]['center_time'] - main_s[1]['center_time']


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
class DoubleS1Scatter_EventPositions(strax.Plugin):
    __version__ = "0.1.0"

    depends_on = ('DoubleS1Scatter_event_basics',)
    provides = 'DoubleS1Scatter_event_positions'
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

        orig_pos = np.vstack([events['x_s2_a'], events['y_s2_a'], z_obs]).T
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
            invalid = np.abs(z_obs) < np.abs(delta_r)# Why??
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
class DoubleS1Scatter_CorrectedAreas(strax.Plugin):
    __version__ = "0.1.0"

    depends_on = ['DoubleS1Scatter_event_basics', 'DoubleS1Scatter_event_positions']
    dtype = [('cs1_a', np.float32, 'Corrected S1_a area (PE)'),
             ('cs1_b', np.float32, 'Corrected S1_b area (PE)'),
             ('cs2_a', np.float32, 'Corrected S2_a area (PE)'),
             ('cs2_b', np.float32, 'Corrected S2_b area (PE)')] + strax.time_fields
    provides = 'DoubleS1Scatter_corrected_areas'

    def setup(self):
        self.s1_map = InterpolatingMap(
            get_resource(self.config['s1_relative_lce_map']))
        self.s2_map = InterpolatingMap(
            get_resource(self.config['s2_relative_lce_map']))
        self.elife = get_elife(self.run_id,self.config['elife_file'])

    def compute(self, events):
        event_positions = np.vstack([events['x'], events['y'], events['z']]).T
        s2_a_positions = np.vstack([events['x_s2_a'], events['y_s2_a']]).T
        s2_b_positions = np.vstack([events['x_s2_b'], events['y_s2_b']]).T
        lifetime_corr = np.exp(
            events['drift_time'] / self.elife)

        return dict(
            time=events['time'],
            endtime=strax.endtime(events),
            cs1_a=events['s1_a_area'] / self.s1_map(event_positions),
            cs1_b=events['s1_b_area'] / self.s1_map(event_positions),
            cs2_a=events['s2_a_area'] * lifetime_corr / self.s2_map(s2_a_positions),
            cs2_b=events['s2_b_area'] * lifetime_corr / self.s2_map(s2_b_positions))


 
class Kr_EventInfo(strax.MergeOnlyPlugin):
    depends_on = ['events',
                  'DoubleS1Scatter_event_basics', 'DoubleS1Scatter_event_positions', 'DoubleS1Scatter_corrected_areas']
    provides = 'DoubleS1Scatter_event_info'
    save_when = strax.SaveWhen.ALWAYS
    
