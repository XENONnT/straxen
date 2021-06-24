import json
import os
import tempfile
import numpy as np
import numba

import strax
import straxen
from straxen.common import pax_file, get_resource, first_sr1_run
export, __all__ = strax.exporter()
from .pulse_processing import  HE_PREAMBLE
from copy import deepcopy

@export
class PeaksExtended(strax.Plugin):
    __version__='0.0.2'
    depends_on=('peaks')
    arrays = ('top','bottom','full')
    provides=('peaks_extended')
    
    def infer_dtype(self):
        n_sum_wv_samples=200
        n_widths=11
        
        dtype=deepcopy(strax.peak_interval_dtype)
        for array in self.arrays:
            dtype += [
            ((f'Integral across channels [PE] for array {array}',
              f'area_{array}'), np.float32),
            ((f'Waveform data in PE/sample for array {array} (not PE/ns!)',
              f'data_{array}'), np.float32, n_sum_wv_samples),
            ((f'Peak widths in range of central area fraction [ns] for array {array}',
              f'width_{array}'), np.float32, n_widths),
            ((f'Peak widths: time between nth and 5th area decile [ns] for array {array}',
              f'area_decile_from_midpoint_{array}'), np.float32, n_widths),
            ((f'Center time for array {array}',
              f'center_time_{array}'), np.int32),
            ]
        return dtype
            
    def compute(self,peaks):
        results = np.zeros(len(peaks),dtype=self.dtype)
        for v in ('dt','length','time'):
            results[f'{v}']=peaks[f'{v}']
        
        results['data_full']=peaks['data']
        results['data_top']=peaks['data_top']
        results['data_bottom']=peaks['data']-peaks['data_top']
            
        for array in self.arrays:
            results[f'area_{array}']=np.sum(results[f'data_{array}'],axis=1)

        center_times=self.compute_widths(results)
        m = peaks['area'] > 0
        results['center_time_top'] = peaks['time']
        results['center_time_bottom'] = peaks['time']
        results['center_time_full'] = peaks['time']

        center_times=self.compute_center_times(results)
        results['center_time_top']+=center_times[:,0]
        results['center_time_bottom']+=center_times[:,1]
        results['center_time_full']+=center_times[:,2]
        return results
        
    def compute_widths(self,peaks):
        """Compute widths in ns at desired area fractions for peaks
        returns (n_peaks, n_widths) array
        """
        if not len(peaks):
            return

        desired_widths = np.linspace(0, 1, len(peaks[0]['width_full']))
        # 0% are width is 0 by definition, and it messes up the calculation below
        desired_widths = desired_widths[1:]

        # Which area fractions do we need times for?
        desired_fr = np.concatenate([0.5 - desired_widths / 2,
                                     0.5 + desired_widths / 2])

        # We lose the 50% fraction with this operation, let's add it back
        desired_fr = np.sort(np.unique(np.append(desired_fr, [0.5])))

        for array in self.arrays:
            fr_times = index_of_fraction(peaks, array, desired_fr)
            fr_times *= peaks['dt'].reshape(-1, 1)

            i = len(desired_fr) // 2
            peaks[f'width_{array}'] = fr_times[:, i:] - fr_times[:, ::-1][:, i:]
            peaks[f'area_decile_from_midpoint_{array}'] = fr_times[:, ::2] - fr_times[:, i].reshape(-1,1)
            
    def compute_center_times(self,peaks):
        result = np.zeros((len(peaks),3), dtype=np.int32)
        for p_i, p in enumerate(peaks):
            for ix,array in enumerate(self.arrays):
                if p[f'area_{array}']==0.0:
                    continue
                t = 0
                for t_i, weight in enumerate(p[f'data_{array}']):
                    t += t_i * p['dt'] * weight
                result[p_i][ix] = t / p[f'area_{array}']
        return result
        

# @numba.njit(cache=True, nogil=True)
def index_of_fraction(peaks, array,fractions_desired):
    """Return the (fractional) indices at which the peaks reach
    fractions_desired of their area
    :param peaks: strax peak(let)s or other data-bearing dtype
    :param fractions_desired: array of floats between 0 and 1
    :returns: (len(peaks), len(fractions_desired)) array of floats
    """
    results = np.zeros((len(peaks), len(fractions_desired)), dtype=np.float32)

    for p_i, p in enumerate(peaks):
        if p[f'area_{array}'] <= 0:
            continue  # TODO: These occur a lot. Investigate!
        results[p_i]=compute_index_of_fraction(p, array, fractions_desired, results[p_i])
    return results


# @numba.jit(nopython=True, nogil=True, cache=True)
def compute_index_of_fraction(peak, array,fractions_desired, result):
    """Store the (fractional) indices at which peak reaches
    fractions_desired of their area in result
    :param peak: single strax peak(let) or other data-bearing dtype
    :param fractions_desired: array of floats between 0 and 1
    :returns: len(fractions_desired) array of floats
    """
    area_tot = peak[f'area_{array}']
    fraction_seen = 0
    current_fraction_index = 0
    needed_fraction = fractions_desired[current_fraction_index]
    for i, x in enumerate(peak['data_'+array][:peak['length']]):
        # How much of the area is in this sample?
        fraction_this_sample = x / area_tot
        # Are we passing any desired fractions in this sample?
        while fraction_seen + fraction_this_sample >= needed_fraction:

            area_needed = area_tot * (needed_fraction - fraction_seen)
            if x != 0:
                result[current_fraction_index] = i + area_needed / x
            else:
                result[current_fraction_index] = i

            # Advance to the next fraction
            current_fraction_index += 1
            if current_fraction_index > len(fractions_desired) - 1:
                break
            needed_fraction = fractions_desired[current_fraction_index]

        if current_fraction_index > len(fractions_desired) - 1:
            break

        # Add this sample's area to the area seen
        fraction_seen += fraction_this_sample

    if needed_fraction == 1:
        # Sometimes floating-point errors prevent the full area
        # from being reached before the waveform ends
        result[-1] = peak['length']
    return result


@export
@strax.takes_config(
    strax.Option('n_top_pmts', default=straxen.n_top_pmts,
                 help="Number of top PMTs"),
    strax.Option('check_peak_sum_area_rtol', default=None, track=False,
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
    __version__ = "0.0.9"
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
            'n_channels'), np.int16),
        (('PMT number which contributes the most PE',
            'max_pmt'), np.int16),
        (('Area of signal in the largest-contributing PMT (PE)',
            'max_pmt_area'), np.float32),
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
        r['range_50p_area'] = p['width'][:, 5]
        r['range_90p_area'] = p['width'][:, 9]
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)
        r['max_pmt_area'] = np.max(p['area_per_channel'], axis=1)
        r['tight_coincidence'] = p['tight_coincidence']

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
        'nn_architecture',
        help='Path to JSON of neural net architecture',
        default_by_run=[
            (0, pax_file('XENON1T_tensorflow_nn_pos_20171217_sr0.json')),
            (first_sr1_run, straxen.aux_repo + '3548132b55f81a43654dba5141366041e1daaf01/strax_files/XENON1T_tensorflow_nn_pos_20171217_sr1_reformatted.json')]),   # noqa
    strax.Option(
        'nn_weights',
        help='Path to HDF5 of neural net weights',
        default_by_run=[
            (0, pax_file('XENON1T_tensorflow_nn_pos_weights_20171217_sr0.h5')),
            (first_sr1_run, pax_file('XENON1T_tensorflow_nn_pos_weights_20171217_sr1.h5'))]),   # noqa
    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area (PE) is less than this',
                 default=10),
    strax.Option('n_top_pmts', default=straxen.n_top_pmts,
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
    strax.Option('min_area_fraction', default=0.5,
                 help='The area of competing peaks must be at least '
                      'this fraction of that of the considered peak'),
    strax.Option('nearby_window', default=int(1e7),
                 help='Peaks starting within this time window (on either side)'
                      'in ns count as nearby.'),
    strax.Option('peak_max_proximity_time', default=int(1e8),
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
