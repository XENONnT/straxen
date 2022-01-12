import json
import os
import tempfile
import numpy as np
import numba
from enum import IntEnum

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
    __version__ = "0.1.0"
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
@strax.takes_config(
    strax.Option(name='pre_s2_area_threshold', default=1000,
                 help='Only take S2s larger than this into account '
                      'when calculating PeakShadow [PE]'),
    strax.Option(name='deltatime_exponent', default=-1.0,
                 help='The exponent of delta t when calculating shadow'),
    strax.Option('time_window_backward', default=int(3e9),
                 help='Search for S2s causing shadow in this time window [ns]'),
    strax.Option(name='electron_drift_velocity',
                 default=('electron_drift_velocity', 'ONLINE', True),
                 help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'),
    strax.Option(name='max_drift_length', default=straxen.tpc_z,
                 help='Total length of the TPC from the bottom of gate to the '
                      'top of cathode wires [cm]'),
    strax.Option(name='exclude_drift_time', default=False,
                 help='Subtract max drift time to avoid peak interference in '
                      'a single event [ns]'))
class PeakShadow(strax.OverlapWindowPlugin):
    """
    This plugin can find and calculate the previous S2 shadow at peak level,
    with time window backward and previous S2 area as options.
    It also gives the area and position information of these previous S2s.
    """

    __version__ = '0.1.0'
    depends_on = ('peak_basics', 'peak_positions')
    provides = 'peak_shadow'
    save_when = strax.SaveWhen.EXPLICIT

    def setup(self):
        self.time_window_backward = self.config['time_window_backward']
        if self.config['exclude_drift_time']:
            electron_drift_velocity = straxen.get_correction_from_cmt(
                self.run_id,
                self.config['electron_drift_velocity'])
            drift_time_max = int(self.config['max_drift_length'] / electron_drift_velocity)
            self.n_drift_time = drift_time_max
        else:
            self.n_drift_time = 0
        self.s2_threshold = self.config['pre_s2_area_threshold']
        self.exponent = self.config['deltatime_exponent']

    def get_window_size(self):
        return 3 * self.config['time_window_backward']

    def infer_dtype(self):
        dtype = [('shadow', np.float32, 'previous s2 shadow [PE/ns]'),
                 ('pre_s2_area', np.float32, 'previous s2 area [PE]'),
                 ('shadow_dt', np.int64, 'time difference to the previous s2 [ns]'),
                 ('pre_s2_x', np.float32, 'x of previous s2 peak causing shadow [cm]'),
                 ('pre_s2_y', np.float32, 'y of previous s2 peak causing shadow [cm]')]
        dtype += strax.time_fields
        return dtype

    def compute(self, peaks):
        roi_shadow = np.zeros(len(peaks), dtype=strax.time_fields)
        roi_shadow['time'] = peaks['center_time'] - self.time_window_backward
        roi_shadow['endtime'] = peaks['center_time'] - self.n_drift_time

        mask_pre_s2 = peaks['area'] > self.s2_threshold
        mask_pre_s2 &= peaks['type'] == 2
        split_peaks = strax.touching_windows(peaks[mask_pre_s2], roi_shadow)
        res = np.zeros(len(peaks), self.dtype)
        res['pre_s2_x'] = np.nan
        res['pre_s2_y'] = np.nan
        if len(peaks):
            self.compute_shadow(peaks, peaks[mask_pre_s2], split_peaks, self.exponent, res)

        res['time'] = peaks['time']
        res['endtime'] = strax.endtime(peaks)
        return res

    @staticmethod
    @numba.njit
    def compute_shadow(peaks, pre_s2_peaks, touching_windows, exponent, res):
        """
        For each peak in peaks, check if there is a shadow-casting S2 peak
        and check if it casts the largest shadow
        """
        for p_i, p_a in enumerate(peaks):
            # reset for every peak
            new_shadow = 0
            s2_indices = touching_windows[p_i]
            for s2_idx in range(s2_indices[0], s2_indices[1]):
                s2_a = pre_s2_peaks[s2_idx]
                if p_a['center_time'] - s2_a['center_time'] <= 0:
                    continue
                new_shadow = s2_a['area'] * (
                        p_a['center_time'] - s2_a['center_time'])**exponent
                if new_shadow > res['shadow'][p_i]:
                    res['shadow'][p_i] = new_shadow
                    res['pre_s2_area'][p_i] = s2_a['area']
                    res['shadow_dt'][p_i] = p_a['center_time'] - s2_a['center_time']
                    res['pre_s2_x'][p_i] = s2_a['x']
                    res['pre_s2_y'][p_i] = s2_a['y']


@export
class VetoPeakTags(IntEnum):
    """Identifies by which detector peak was tagged.
    """
    # Peaks are not inside any veto interval
    NO_VETO = 0
    # Peaks are inside a veto interval issued by:
    NEUTRON_VETO = 1
    MUON_VETO = 2
    BOTH = 3


@export
class PeakVetoTagging(strax.Plugin):
    """
    Plugin which tags S1 peaks according to  muon and neutron-vetos.
    Tagging S2s is does not make sense as they occur with a delay.
    However, we compute for both S1/S2 the time delay to the closest veto
    region.

        * untagged: 0
        * neutron-veto: 1
        * muon-veto: 2
        * both vetos: 3
    """
    __version__ = '0.0.1'
    depends_on = ('peak_basics', 'veto_regions_nv', 'veto_regions_mv')
    provides = ('peak_veto_tags')
    save_when = strax.SaveWhen.TARGET

    dtype = strax.time_fields + [
        ('veto_tag', np.int8,
         'Veto tag for S1 peaks. unatagged: 0, nveto: 1, mveto: 2, both: 3'),
        ('time_to_closest_veto', np.int64, 'Time to closest veto interval boundary in ns (can be '
                                           'negative if closest boundary comes before peak.). ')
    ]

    def get_time_difference(self, peaks, veto_regions_nv, veto_regions_mv):
        """
        Computes time differences to closest nv/mv veto signal.

        It might be that neutron-veto and muon-veto signals overlap
        Hence we compute first the individual time differences to the
        corresponding vetos and keep afterwards the smallest ones.
        """
        dt_nv = get_time_to_closest_veto(peaks, veto_regions_nv)
        dt_mv = get_time_to_closest_veto(peaks, veto_regions_mv)

        dts = np.transpose([dt_nv, dt_mv])
        ind_axis1 = np.argmin(np.abs(dts), axis=1)
        return self._get_smallest_value(dts, ind_axis1)

    @staticmethod
    @numba.njit(cache=True, nogil=True)
    def _get_smallest_value(time_differences, index):
        res = np.zeros(len(time_differences), np.int64)
        for res_ind, (ind, dt) in enumerate(zip(index, time_differences)):
            res[res_ind] = dt[ind]
        return res

    def compute(self, peaks, veto_regions_nv, veto_regions_mv):
        touching_mv = strax.touching_windows(peaks, veto_regions_mv)
        touching_nv = strax.touching_windows(peaks, veto_regions_nv)

        tags = np.zeros(len(peaks))
        tags = tag_peaks(tags, touching_nv, straxen.VetoPeakTags.NEUTRON_VETO)
        tags = tag_peaks(tags, touching_mv, straxen.VetoPeakTags.MUON_VETO)

        dt = self.get_time_difference(peaks, veto_regions_nv, veto_regions_mv)
        return {'time': peaks['time'],
                'endtime': strax.endtime(peaks),
                'veto_tag': tags,
                'time_to_closest_veto': dt,
                }


@numba.njit(cache=True, nogil=True)
def tag_peaks(tags, touching_windows, tag_number):
    """Tags every peak which are within the corresponding touching window
    with the defined tag number.

    :param tags: numpy.array in which the tags should be stored. Should
        be of length peaks.
    :param touching_windows: Start/End index of tags to be set to tag
        value.
    :param tag_number: integer representing the tag.
    :return: Updated tags.
    """
    pre_tags = np.zeros(len(tags), dtype=np.int8)
    for start, end in touching_windows:
        pre_tags[start:end] = tag_number
    tags += pre_tags
    return tags


def get_time_to_closest_veto(peaks, veto_intervals):
    """Computes time difference between peak and closest veto interval.

    The time difference is always computed from peaks-time field to
    the time or endtime of the veto_interval depending on which distance
    is smaller.
    """
    vetos = np.zeros(len(veto_intervals)+2, strax.time_fields)
    vetos[1:-1]['time'] = veto_intervals['time']
    vetos[1:-1]['endtime'] = strax.endtime(veto_intervals)
    vetos[-1]['time'] = straxen.INFINITY_64BIT_SIGNED
    vetos[-1]['endtime'] = straxen.INFINITY_64BIT_SIGNED
    vetos[0]['time'] = -straxen.INFINITY_64BIT_SIGNED
    vetos[0]['endtime'] = -straxen.INFINITY_64BIT_SIGNED
    return _get_time_to_closest_veto(peaks, vetos)


@numba.njit(cache=True, nogil=True)
def _get_time_to_closest_veto(peaks, vetos):
    res = np.zeros(len(peaks), dtype=np.int64)
    veto_index = 0
    for ind, p in enumerate(peaks):
        for veto_index in range(veto_index, len(vetos)):
            if veto_index+1 == len(vetos):
                # If we reach here all future peaks are closest to last veto:
                res[ind] = np.abs(vetos[-1]['time'] - p['time'])
                break

            # Current interval can be before or after current peak, hence
            # we have to check which distance is smaller.
            dt_current_veto = min(np.abs(vetos[veto_index]['time'] - p['time']),
                                  np.abs(vetos[veto_index]['endtime'] - p['time'])
                                  )
            # Next interval is always further in the future as the
            # current one, hence we only have to compute distance with "time".
            dt_next_veto = np.abs(vetos[veto_index+1]['time'] - p['time'])

            # Next veto is closer so we have to repeat in case next + 1
            # is even closer.
            if dt_current_veto >= dt_next_veto:
                veto_index += 1
                continue

            # Now compute time difference for real:
            dt_time = vetos[veto_index]['time'] - p['time']
            dt_endtime = vetos[veto_index]['endtime'] - p['time']

            if np.abs(dt_time) < np.abs(dt_endtime):
                res[ind] = dt_time
            else:
                res[ind] = dt_endtime
            break

    return res
