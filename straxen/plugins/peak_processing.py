import json
import os
import tempfile
import numpy as np
import numba
from enum import IntEnum
from scipy.stats import halfcauchy

import strax
import straxen
from straxen.common import pax_file, get_resource, first_sr1_run
export, __all__ = strax.exporter()
from .pulse_processing import HE_PREAMBLE


@export
@strax.takes_config(
    strax.Option('n_top_pmts', default=straxen.n_top_pmts, infer_type=False,
                 help="Number of top PMTs"),
    strax.Option('check_peak_sum_area_rtol', default=None, track=False, infer_type=False,
                 help="Check if the sum area and the sum of area per "
                      "channel are the same. If None, don't do the "
                      "check. To perform the check, set to the desired "
                      " rtol value used e.g. '1e-4' (see np.isclose)."),
)
class PeakBasics(strax.Plugin):
    """
    Compute the basic peak-properties, thereby dropping structured
    arrays.
    NB: This plugin can therefore be loaded as a pandas DataFrame.
    """
    __version__ = "0.1.1"
    parallel = True
    depends_on = ('peaks',)
    provides = 'peak_basics'
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
            'n_hits'), np.int32),
        (('Number of hits contributing at least one sample to the peak',
            'n_channels'), np.int16),
        (('PMT number which contributes the most PE',
            'max_pmt'), np.int16),
        (('Area of signal in the largest-contributing PMT (PE)',
            'max_pmt_area'), np.float32),
        (('Total number of saturated channels',
          'n_saturated_channels'), np.int16),
        (('Width (in ns) of the central 50% area of the peak',
            'range_50p_area'), np.float32),
        (('Width (in ns) of the central 90% area of the peak',
            'range_90p_area'), np.float32),
        (('Fraction of area seen by the top array '
          '(NaN for peaks with non-positive area)',
            'area_fraction_top'), np.float32),
        (('Length of the peak waveform in samples',
          'length'), np.int32),
        (('Time resolution of the peak waveform in ns',
          'dt'), np.int16),
        (('Time between 10% and 50% area quantiles [ns]',
          'rise_time'), np.float32),
        (('Hits within tight range of mean',
          'tight_coincidence'), np.int16),
        (('PMT channel within tight range of mean',
          'tight_coincidence_channel'), np.int16),
        (('Classification of the peak(let)',
          'type'), np.int8)
    ]

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), self.dtype)
        for q in 'time length dt area type'.split():
            r[q] = p[q]
        r['endtime'] = p['time'] + p['dt'] * p['length']
        r['n_channels'] = (p['area_per_channel'] > 0).sum(axis=1)
        r['n_hits'] = p['n_hits']
        r['range_50p_area'] = p['width'][:, 5]
        r['range_90p_area'] = p['width'][:, 9]
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)
        r['max_pmt_area'] = np.max(p['area_per_channel'], axis=1)
        r['tight_coincidence'] = p['tight_coincidence']
        r['n_saturated_channels'] = p['n_saturated_channels']

        n_top = self.config['n_top_pmts']
        area_top = p['area_per_channel'][:, :n_top].sum(axis=1)
        # Recalculate to prevent numerical inaccuracy #442
        area_total = p['area_per_channel'].sum(axis=1)
        # Negative-area peaks get NaN AFT
        m = p['area'] > 0
        r['area_fraction_top'][m] = area_top[m]/area_total[m]
        r['area_fraction_top'][~m] = float('nan')
        r['rise_time'] = -p['area_decile_from_midpoint'][:, 1]

        if self.config['check_peak_sum_area_rtol'] is not None:
            self.check_area(area_total, p, self.config['check_peak_sum_area_rtol'])
        # Negative or zero-area peaks have centertime at startime
        r['center_time'] = p['time']
        r['center_time'][m] += self.compute_center_times(peaks[m])
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

    @staticmethod
    def check_area(area_per_channel_sum, peaks, rtol) -> None:
        """
        Check if the area of the sum-wf is the same as the total area
            (if the area of the peak is positively defined).

        :param area_per_channel_sum: the summation of the
            peaks['area_per_channel'] which will be checked against the
             values of peaks['area'].
        :param peaks: array of peaks.
        :param rtol: relative tolerance for difference between
            area_per_channel_sum and peaks['area']. See np.isclose.
        :raises: ValueError if the peak area and the area-per-channel
            sum are not sufficiently close
        """
        positive_area = peaks['area'] > 0
        if not np.sum(positive_area):
            return

        is_close = np.isclose(area_per_channel_sum[positive_area],
                              peaks[positive_area]['area'],
                              rtol=rtol,
                             )

        if not is_close.all():
            for peak in peaks[positive_area][~is_close]:
                print('bad area')
                strax.print_record(peak)

            p_i = np.where(~is_close)[0][0]
            peak = peaks[positive_area][p_i]
            area_fraction_off = 1 - area_per_channel_sum[positive_area][p_i] / peak['area']
            message = (f'Area not calculated correctly, it\'s '
                       f'{100*area_fraction_off} % off, time: {peak["time"]}')
            raise ValueError(message)


@export
class PeakBasicsHighEnergy(PeakBasics):
    __doc__ = HE_PREAMBLE + PeakBasics.__doc__
    __version__ = '0.0.2'
    depends_on = 'peaks_he'
    provides = 'peak_basics_he'
    child_ends_with = '_he'

    def compute(self, peaks_he):
        return super().compute(peaks_he)


@export
@strax.takes_config(
    strax.Option(
        'nn_architecture', infer_type=False,
        help='Path to JSON of neural net architecture',
        default_by_run=[
            (0, pax_file('XENON1T_tensorflow_nn_pos_20171217_sr0.json')),
            (first_sr1_run, straxen.aux_repo + '3548132b55f81a43654dba5141366041e1daaf01/strax_files/XENON1T_tensorflow_nn_pos_20171217_sr1_reformatted.json')]),   # noqa
    strax.Option(
        'nn_weights', infer_type=False,
        help='Path to HDF5 of neural net weights',
        default_by_run=[
            (0, pax_file('XENON1T_tensorflow_nn_pos_weights_20171217_sr0.h5')),
            (first_sr1_run, pax_file('XENON1T_tensorflow_nn_pos_weights_20171217_sr1.h5'))]),   # noqa
    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area (PE) is less than this',
                 default=10,  infer_type=False,),
    strax.Option('n_top_pmts', default=straxen.n_top_pmts, infer_type=False,
                 help="Number of top PMTs")
)
class PeakPositions1T(strax.Plugin):
    """Compute the S2 (x,y)-position based on a neural net."""
    dtype = [('x', np.float32,
              'Reconstructed S2 X position (cm), uncorrected'),
             ('y', np.float32,
              'Reconstructed S2 Y position (cm), uncorrected')
             ] + strax.time_fields
    depends_on = ('peaks',)
    provides = "peak_positions"

    # Parallelization doesn't seem to make it go faster
    # Is there much pure-python stuff in tensorflow?
    # Process-level paralellization might work, but you'd have to do setup
    # in each process, which probably negates the benefits,
    # except for huge chunks
    parallel = False

    __version__ = '0.1.1'

    def setup(self):
        import tensorflow as tf
        keras = tf.keras
        nn_conf = get_resource(self.config['nn_architecture'], fmt='json')
        # badPMTList was inserted by a very clever person into the keras json
        # file. Let's delete it to prevent future keras versions from crashing.
        # Do NOT try `del nn_conf['badPMTList']`! See get_resource docstring
        # for the gruesome details.
        bad_pmts = nn_conf['badPMTList']
        nn = keras.models.model_from_json(json.dumps({
            k: v
            for k, v in nn_conf.items()
            if k != 'badPMTList'}))
        self.pmt_mask = ~np.in1d(np.arange(self.config['n_top_pmts']),
                                 bad_pmts)

        # Keras needs a file to load its weights. We can't put the load
        # inside the context, then it would break on Windows,
        # because there temporary files cannot be opened again.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(get_resource(self.config['nn_weights'],
                                 fmt='binary'))
            fname = f.name
        nn.load_weights(fname)
        os.remove(fname)
        self.nn = nn

    def compute(self, peaks):
        result = np.ones(len(peaks), dtype=self.dtype)
        result['time'], result['endtime'] = peaks['time'], strax.endtime(peaks)
        result['x'] *= float('nan')
        result['y'] *= float('nan')

        # Keep large peaks only
        peak_mask = peaks['area'] > self.config['min_reconstruction_area']
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result

        # Input: normalized hitpatterns in good top PMTs
        _in = peaks['area_per_channel'][peak_mask, :]
        _in = _in[:, :self.config['n_top_pmts']][:, self.pmt_mask]
        with np.errstate(divide='ignore', invalid='ignore'):
            _in /= _in.sum(axis=1).reshape(-1, 1)

        # Output: positions in mm (unfortunately), so convert to cm
        _out = self.nn.predict(_in) / 10

        # Set output in valid rows. Do NOT try result[peak_mask]['x']
        # unless you want all NaN positions (boolean masks make a copy unless
        # they are used as the last index)
        result['x'][peak_mask] = _out[:, 0]
        result['y'][peak_mask] = _out[:, 1]
        return result


@export
@strax.takes_config(
    strax.Option('min_area_fraction', default=0.5, infer_type=False,
                 help='The area of competing peaks must be at least '
                      'this fraction of that of the considered peak'),
    strax.Option('nearby_window', default=int(1e7), infer_type=False,
                 help='Peaks starting within this time window (on either side)'
                      'in ns count as nearby.'),
    strax.Option('peak_max_proximity_time', default=int(1e8), infer_type=False,
                 help='Maximum value for proximity values such as '
                      't_to_next_peak [ns]'))
class PeakProximity(strax.OverlapWindowPlugin):
    """
    Look for peaks around a peak to determine how many peaks are in
    proximity (in time) of a peak.
    """
    depends_on = ('peak_basics',)
    dtype = [
        ('n_competing', np.int32,
         'Number of nearby larger or slightly smaller peaks'),
        ('n_competing_left', np.int32,
         'Number of larger or slightly smaller peaks left of the main peak'),
        ('t_to_prev_peak', np.int64,
         'Time between end of previous peak and start of this peak [ns]'),
        ('t_to_next_peak', np.int64,
         'Time between end of this peak and start of next peak [ns]'),
        ('t_to_nearest_peak', np.int64,
         'Smaller of t_to_prev_peak and t_to_next_peak [ns]')
    ] + strax.time_fields

    __version__ = '0.4.0'

    def get_window_size(self):
        return self.config['peak_max_proximity_time']

    def compute(self, peaks):
        windows = strax.touching_windows(peaks, peaks,
                                         window=self.config['nearby_window'])
        n_left, n_tot = self.find_n_competing(
            peaks,
            windows,
            fraction=self.config['min_area_fraction'])

        t_to_prev_peak = (
                np.ones(len(peaks), dtype=np.int64)
                * self.config['peak_max_proximity_time'])
        t_to_prev_peak[1:] = peaks['time'][1:] - peaks['endtime'][:-1]

        t_to_next_peak = t_to_prev_peak.copy()
        t_to_next_peak[:-1] = peaks['time'][1:] - peaks['endtime'][:-1]

        return dict(
            time=peaks['time'],
            endtime=strax.endtime(peaks),
            n_competing=n_tot,
            n_competing_left=n_left,
            t_to_prev_peak=t_to_prev_peak,
            t_to_next_peak=t_to_next_peak,
            t_to_nearest_peak=np.minimum(t_to_prev_peak, t_to_next_peak))

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def find_n_competing(peaks, windows, fraction):
        n_left = np.zeros(len(peaks), dtype=np.int32)
        n_tot = n_left.copy()
        areas = peaks['area']

        for i, peak in enumerate(peaks):
            left_i, right_i = windows[i]
            threshold = areas[i] * fraction
            n_left[i] = np.sum(areas[left_i:i] > threshold)
            n_tot[i] = n_left[i] + np.sum(areas[i + 1:right_i] > threshold)

        return n_left, n_tot

@export
class PeakShadow(strax.OverlapWindowPlugin):
    """
    This plugin can find and calculate the time & position shadow
    from previous peaks in time.
    It also gives the area and (x,y) of the previous peaks.
    References:
        * v0.1.3 reference: xenon:xenonnt:ac:prediction:shadow_ambience
    """

    __version__ = '0.1.3'
    depends_on = ('peak_basics', 'peak_positions')
    provides = 'peak_shadow'
    save_when = strax.SaveWhen.EXPLICIT

    shadow_time_window_backward = straxen.URLConfig(
        default=int(1e9),
        type=int,
        track=True,
        help='Search for peaks casting time & position shadow in this time window [ns]'
    )

    shadow_threshold = straxen.URLConfig(
        default={'s1_time_shadow' : 1e3, 
                 's2_time_shadow' : 1e4, 
                 's2_position_shadow' : 1e4},
        type=dict,
        track=True,
        help='Only take S1/S2s larger than this into account when calculating Shadow [PE]'
    )

    shadow_deltatime_exponent = straxen.URLConfig(
        default=-1.0,
        type=float,
        track=True,
        help='The exponent of delta t when calculating shadow'
    )

    shadow_sigma_and_baseline = straxen.URLConfig(
        default=[15.220, 0.036],
        type=list,
        track=True,
        help='Fitted position correlation sigma[cm*PE^0.5] and baseline[cm] using in position shadow'
    )

    def get_window_size(self):
        return 10 * self.config['shadow_time_window_backward']

    @property
    def shadtype(self):
        # Shadow related features shared by previous S1 & S2
        dtype = []
        dtype += [('shadow', np.float32), ('dt', np.int64)]
        dtype += [('x', np.float32), ('y', np.float32)]
        dtype += [('nearest_dt', np.int64)]
        return dtype

    def infer_dtype(self):
        s1_time_shadow_dtype = []
        s2_time_shadow_dtype = []
        s2_position_shadow_dtype = []
        nearest_dtype = []
        # We have time shadow(S2/dt) and position shadow(S2/dt*p(s))
        # previous S1 can only cast time shadow, previous S2 can cast both time & position shadow
        for key, dtype in zip(['s1_time_shadow', 's2_time_shadow', 's2_position_shadow'], 
                              [s1_time_shadow_dtype, s2_time_shadow_dtype, s2_position_shadow_dtype]):
            type_str, tp_desc, _ = key.split('_')
            dtype.append(((f'previous large {type_str} casted largest {tp_desc} shadow [PE/ns]', 
                           f'shadow_{key}'), np.float32))
            dtype.append(((f'time difference to the previous large {type_str} peak casting largest {tp_desc} shadow [ns]', 
                           f'dt_{key}'), np.int64))
            # Only previous S2 peaks have (x,y)
            if 's2' in key:
                dtype.append(((f'x of previous large s2 peak casting largest {tp_desc} shadow [cm]', 
                               f'x_{key}'), np.float32))
                dtype.append(((f'y of previous large s2 peak casting largest {tp_desc} shadow [cm]', 
                               f'y_{key}'), np.float32))
            # Only time shadow gives the nearest large peak
            if 'time' in key:
                dtype.append(((f'time difference to the nearest previous large {type_str}', 
                               f'nearest_dt_{type_str}'), np.int64))
        # Also record the PDF of HalfCauchy when calculating S2 position shadow
        s2_position_shadow_dtype.append((('PDF describing correlation to the previous large s2', 
                                          'pdf_s2_position_shadow'), np.float32))

        dtype = s1_time_shadow_dtype + s2_time_shadow_dtype + s2_position_shadow_dtype + nearest_dtype + strax.time_fields
        return dtype

    @property
    def shadowdtype(self):
        dtype = []
        dtype += [('shadow', np.float32), ('dt', np.int64)]
        dtype += [('x', np.float32), ('y', np.float32)]
        dtype += [('nearest_dt', np.int64)]
        return dtype

    def compute(self, peaks):
        return self.compute_shadow(peaks, peaks)

    def compute_shadow(self, peaks, current_peak):
        # 1. Define time window for each peak, we will find previous peaks within these time windows
        roi_shadow = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi_shadow['time'] = current_peak['center_time'] - self.config['shadow_time_window_backward']
        roi_shadow['endtime'] = current_peak['center_time']

        # 2. Calculate S2 position shadow, S2 time shadow, and S1 time shadow
        result = np.zeros(len(current_peak), self.dtype)
        for key in ['s2_position_shadow', 's2_time_shadow', 's1_time_shadow']:
            is_position = 'position' in key
            type_str = key.split('_')[0]
            stype = 2 if 's2' in key else 1
            mask_pre = (peaks['type'] == stype) & (peaks['area'] > self.config['shadow_threshold'][key])
            split_peaks = strax.touching_windows(peaks[mask_pre], roi_shadow)
            array = np.zeros(len(current_peak), np.dtype(self.shadowdtype))

            # Initialization
            array['x'] = np.nan
            array['y'] = np.nan
            array['dt'] = self.config['shadow_time_window_backward']
            # The default value for shadow is set to be the lowest possible value
            if 'time' in key:
                array['shadow'] = self.config['shadow_threshold'][key] * array['dt'] ** self.config['shadow_deltatime_exponent']
            else:
                array['shadow'] = 0
            array['nearest_dt'] = self.config['shadow_time_window_backward']

            # Calculating shadow, the Major of the plugin. Only record the previous peak casting the largest shadow
            if len(current_peak):
                self.peaks_shadow(current_peak, 
                                  peaks[mask_pre], 
                                  split_peaks, 
                                  self.config['shadow_deltatime_exponent'], 
                                  array, 
                                  is_position, 
                                  self.getsigma(self.config['shadow_sigma_and_baseline'], current_peak['area']))
            
            # Fill results
            names = ['shadow', 'dt']
            if 's2' in key: # Only previous S2 peaks have (x,y)
                names += ['x', 'y']
            if 'time' in key: # Only time shadow gives the nearest large peak
                names += ['nearest_dt']
            for name in names:
                if name == 'nearest_dt':
                    result[f'{name}_{type_str}'] = array[name]
                else:
                    result[f'{name}_{key}'] = array[name]

        distance = np.sqrt((result[f'x_s2_position_shadow'] - current_peak['x']) ** 2 + 
                           (result[f'y_s2_position_shadow'] - current_peak['y']) ** 2)
        # If distance is NaN, set largest distance
        distance = np.where(np.isnan(distance), 2 * straxen.tpc_r, distance)
        # HalfCauchy PDF when calculating S2 position shadow
        result['pdf_s2_position_shadow'] = halfcauchy.pdf(distance, 
                                                          scale=self.getsigma(self.config['shadow_sigma_and_baseline'], current_peak['area']))

        # 6. Set time and endtime for peaks
        result['time'] = current_peak['time']
        result['endtime'] = strax.endtime(current_peak)
        return result

    @staticmethod
    def getsigma(sigma_and_baseline, s2):
        # The parameter of HalfCauchy, which is a function of S2 area
        return sigma_and_baseline[0] / np.sqrt(s2) + sigma_and_baseline[1]

    @staticmethod
    @numba.njit
    def peaks_shadow(peaks, pre_peaks, 
                     touching_windows, exponent, 
                     result, pos_corr, sigmas=None):
        """
        For each peak in peaks, check if there is a shadow-casting peak
        and check if it casts the largest shadow
        """
        for p_i, (suspicious_peak, sigma) in enumerate(zip(peaks, sigmas)):
            # casting_peak is the previous large peak casting shadow
            # suspicious_peak is the suspicious peak which in shadow from casting_peak
            indices = touching_windows[p_i]
            for idx in range(indices[0], indices[1]):
                casting_peak = pre_peaks[idx]
                dt = suspicious_peak['center_time'] - casting_peak['center_time']
                if dt <= 0:
                    continue
                # First we record the time difference to the nearest previous peak
                result['nearest_dt'][p_i] = min(result['nearest_dt'][p_i], dt)
                # Calculate time shadow
                new_shadow = casting_peak['area'] * dt ** exponent
                if pos_corr:
                    # Calculate position shadow which is time shadow with a HalfCauchy PDF multiplier
                    distance = distance_in_xy(suspicious_peak, casting_peak)
                    distance = np.where(np.isnan(distance), 2 * straxen.tpc_r, distance)
                    new_shadow *= 2 / (np.pi * sigma * (1 + (distance / sigma) ** 2))
                # Only the previous peak with largest shadow is recorded
                if new_shadow > result['shadow'][p_i]:
                    result['shadow'][p_i] = new_shadow
                    result['x'][p_i] = casting_peak['x']
                    result['y'][p_i] = casting_peak['y']
                    result['dt'][p_i] = suspicious_peak['center_time'] - casting_peak['center_time']


@export
class PeakAmbience(strax.OverlapWindowPlugin):
    """
    Calculate Ambience of peaks.
    Features are the number of lonehits, small S0, S1, S2 in a time window before peaks,
    and the number of small S2 in circle near the S2 peak in a time window.
    References:
        * v0.0.6 reference: xenon:xenonnt:ac:prediction:shadow_ambience
    """
    __version__ = '0.0.6'
    depends_on = ('lone_hits', 'peak_basics', 'peak_positions')
    provides = 'peak_ambience'
    data_kind = 'peaks'
    save_when = strax.SaveWhen.EXPLICIT

    ambience_time_window_backward = straxen.URLConfig(
        default=int(2e6),
        type=int,
        track=True,
        help='Search for ambience in this time window [ns]'
    )

    ambience_divide_t = straxen.URLConfig(
        default=False,
        type=bool,
        track=True,
        help='Whether to divide area by time difference of ambience creating peak to current peak'
    )

    ambience_divide_r = straxen.URLConfig(
        default=False,
        type=bool,
        track=True,
        help='Whether to divide area by radial distance of ambience creating peak to current peak'
    )

    ambient_radius = straxen.URLConfig(
        default=6.7,
        type=float,
        track=True,
        help='Search for ambience in this radius [cm]'
    )

    ambience_area_parameters = straxen.URLConfig(
        default=(5, 60, 60), 
        type=(list, tuple), 
        track=True,
        help='The upper limit of S0, S1, S2 area to be counted'
    )

    def get_window_size(self):
        return 10 * self.config['ambience_time_window_backward']

    @property
    def origin_dtype(self):
        return ['lh_before', 's0_before', 's1_before', 's2_before', 's2_near']

    def infer_dtype(self):
        dtype = []
        for ambience in self.origin_dtype:
            dtype += [((f"Number of small {' '.join(ambience.split('_'))} a peak", 
                        f'n_{ambience}'), np.int16), 
                      ((f"Area sum of small {' '.join(ambience.split('_'))} a peak", 
                        f's_{ambience}'), np.float32)]
        dtype += strax.time_fields
        return dtype

    def compute(self, lone_hits, peaks):
        return self.compute_ambience(lone_hits, peaks, peaks)

    def compute_ambience(self, lone_hits, peaks, current_peak):
        # 1. Initialization
        result = np.zeros(len(current_peak), self.dtype)
        num_array = np.zeros(len(current_peak), dtype=np.int16)
        sum_array = np.zeros(len(current_peak), dtype=np.float32)

        # 2. Define time window for each peak, we will find small peaks & lone hits within these time windows
        roi = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi['time'] = current_peak['center_time'] - self.config['ambience_time_window_backward']
        roi['endtime'] = current_peak['center_time']

        # 3. Calculate number and area sum of lonehits before a peak
        touching_windows = strax.touching_windows(lone_hits, roi)
        # Calculating ambience
        self.lonehits_ambience(current_peak, 
                               lone_hits, 
                               touching_windows, 
                               result['n_lh_before'],
                               result['s_lh_before'],
                               self.config['ambience_divide_t'])

        # 4. Calculate number and area sum of small S0, S1, S2 before a peak
        radius = -1
        for stype, area in zip([0, 1, 2], 
                                         self.config['ambience_area_parameters']):
            num_array = np.zeros(len(current_peak), dtype=np.int16)
            sum_array = np.zeros(len(current_peak), dtype=np.float32)
            mask_pre = (peaks['type'] == stype) & (peaks['area'] < area)
            touching_windows = strax.touching_windows(peaks[mask_pre], roi)
            # Calculating ambience
            self.peaks_ambience(current_peak, 
                                peaks[mask_pre], 
                                touching_windows, 
                                radius, 
                                result[f'n_s{stype}_before'],
                                result[f's_s{stype}_before'],
                                self.config['ambience_divide_t'], 
                                self.config['ambience_divide_r'])

        # 5. Calculate number and area sum of small S2 near(in (x,y) space) a S2 peak
        num_array = np.zeros(len(current_peak), dtype=np.int16)
        sum_array = np.zeros(len(current_peak), dtype=np.float32)
        mask_pre = (peaks['type'] == 2) & (peaks['area'] < self.config['ambience_area_parameters'][2])
        touching_windows = strax.touching_windows(peaks[mask_pre], roi)
        # Calculating ambience
        self.peaks_ambience(current_peak, 
                            peaks[mask_pre], 
                            touching_windows, 
                            self.config['ambient_radius'], num_array, 
                            sum_array, 
                            self.config['ambience_divide_t'], 
                            self.config['ambience_divide_r'])
        result['n_s2_near'] = num_array
        result['s_s2_near'] = sum_array

        # 6. Set time and endtime for peaks
        result['time'] = current_peak['time']
        result['endtime'] = strax.endtime(current_peak)
        return result

    @staticmethod
    @numba.njit
    def lonehits_ambience(peaks, pre_hits, 
                          touching_windows, num_array, 
                          sum_array, ambience_divide_t):
        # Function to find lonehits before a peak
        # creating_hit is the lonehit creating ambience
        # suspicious_peak is the suspicious peak in the ambience created by creating_hit
        for p_i, suspicious_peak in enumerate(peaks):
            indices = touching_windows[p_i]
            for idx in range(indices[0], indices[1]):
                creating_hit = pre_hits[idx]
                dt = suspicious_peak['center_time'] - creating_hit['time']
                if (dt <= 0) or (creating_hit['area'] <= 0):
                    continue
                num_array[p_i] += 1
                # Sometimes we may interested in sum of area / dt
                if ambience_divide_t:
                    sum_array[p_i] += creating_hit['area'] / dt
                else:
                    sum_array[p_i] += creating_hit['area']

    @staticmethod
    @numba.njit
    def peaks_ambience(peaks, pre_peaks, 
                       touching_windows, ambient_radius, 
                       num_array, sum_array, 
                       ambience_divide_t, ambience_divide_r):
        # Function to find S0, S1, S2 before or near a peak
        # creating_peak is the peak creating ambience
        # suspicious_peak is the suspicious peak in the ambience created by creating_peak
        for p_i, suspicious_peak in enumerate(peaks):
            indices = touching_windows[p_i]
            for idx in range(indices[0], indices[1]):
                creating_peak = pre_peaks[idx]
                r = distance_in_xy(suspicious_peak, creating_peak)
                dt = suspicious_peak['center_time'] - creating_peak['center_time']
                if dt <= 0:
                    continue
                if (ambient_radius < 0) or (r <= ambient_radius):
                    num_array[p_i] += 1
                    # Sometimes we may interested in sum of area / dt
                    if ambience_divide_t:
                        sum_array[p_i] += creating_peak['area'] / dt
                    else:
                        sum_array[p_i] += creating_peak['area']
                    # Sometimes we may interested in sum of area / r^2
                    if ambience_divide_r and ambient_radius > 0:
                        sum_array[p_i] /= r**2


@numba.njit
def distance_in_xy(peak_a, peak_b):
    """Distance between S2s in (x,y)"""
    return np.sqrt((peak_a['x'] - peak_b['x']) ** 2 + 
                   (peak_a['y'] - peak_b['y']) ** 2)
