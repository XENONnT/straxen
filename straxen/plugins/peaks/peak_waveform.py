from .peak_basics import PeakBasics
import straxen
import strax
import numpy as np
import numba
from ..peaklets.peaklets import hit_max_sample

export, __all__ = strax.exporter()


class HitsS1(strax.OverlapWindowPlugin):
    """
    Return timing and area for single hits in S1s.
    Reconstruct hits for S1s with area smaller than 100PE
    """
    __version__ = "0.0.0"
    parallel = True
    depends_on = ('peak_basics', 'records',)
    provides = 'hits_s1'
    data_kind = 'hits'
    save_when = strax.SaveWhen.ALWAYS

    peak_waveform_max_s1_area = straxen.URLConfig(default=100, infer_type=False,
                                                  help='Maximum area of S1 we do hit analysis')

    gain_model = straxen.URLConfig(infer_type=False,
                                   help='PMT gain model. Specify as (model_type, model_config)',
                                   )

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    hit_min_amplitude = straxen.URLConfig(
        track=True, infer_type=False,
        default='cmt://hit_thresholds_tpc?version=ONLINE&run_id=plugin.run_id',
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_tpc_pmts, or a number,'
             'or a string like "pmt_commissioning_initial" which means calling'
             'hitfinder_thresholds.py'
             'or a tuple like (correction=str, version=str, nT=boolean),'
             'which means we are using cmt.'
    )

    def setup(self):
        self.drift_time_max = int(straxen.tpc_z / self.electron_drift_velocity)
        self.to_pe = self.gain_model
        self.hit_thresholds = self.hit_min_amplitude
        return

    def infer_dtype(self):
        self.hits1_dtype = [
            # name                dtype       comment
            ('hits_max_time', np.int64, 'Hits max_time'),
            ('hits_area', np.float64, 'Hits area'),
            ('hits_channel', np.int16, 'Hits channel'),
            ('hits_height', np.float64, 'Hits height'),
        ]
        dtype = strax.time_fields + self.hits1_dtype
        return dtype

    def get_window_size(self):
        # return 1000  # ns
        return self.drift_time_max

    def reconstruct_hits_from_records(self, r):
        # See straxen.plugins.peaklet_processing for reference
        # We trick the lone_hit integration to get the area of hits
        hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)
        hits = hits[self.to_pe[hits['channel']] != 0]
        hits = strax.sort_by_time(hits)
        strax.integrate_lone_hits(lone_hits=hits, records=r,
                                  peaks=np.zeros(0, dtype=strax.time_fields),
                                  save_outside_hits=(30, 200),
                                  n_channels=len(self.to_pe))
        hitlet_time_shift = (hits['left'] - hits['left_integration']) * hits['dt']
        hits['time'] = hits['time'] - hitlet_time_shift
        hits['length'] = (hits['right_integration'] - hits['left_integration'])
        hits = strax.sort_by_time(hits)
        hit_max_times = np.sort(
            hits['time']
            + hits['dt'] * hit_max_sample(r, hits)
            + hitlet_time_shift
        )
        hitlets = np.zeros(len(hits), dtype=self.dtype)
        hitlets['hits_channel'] = np.zeros(len(hits)) * (-1)
        for dtype in ['channel', 'area', 'height']:
            hitlets[f'hits_{dtype}'] = hits[dtype]
        hitlets['hits_max_time'] = hit_max_times
        hitlets['time'] = hits['time']
        hitlets['endtime'] = strax.endtime(hits)
        return hitlets

    def compute(self, peaks, records):
        # Perform waveform analysis only on records contained in small S1s
        mask = (peaks['type'] == 1) & (peaks['area'] <= self.peak_waveform_max_s1_area)
        records_in_s1 = strax.split_touching_windows(records, peaks[mask])
        if records_in_s1:
            # if there is any records contained in S1
            records_in_s1 = np.concatenate(records_in_s1)
            hitlets = self.reconstruct_hits_from_records(records_in_s1)
            return hitlets


class PeakWaveformS1(strax.OverlapWindowPlugin):
    """
    Return timing and area for single hits in S1s.
    Reconstruct hits for S1s with area smaller than 100PE
    Put hit shadow parameter for hits
    """
    __version__ = "0.0.0"
    parallel = True
    depends_on = ('peak_basics', 'peaks', 'hits_s1',)
    provides = 'peak_waveform_s1'
    data_kind = 'peaks'
    save_when = strax.SaveWhen.ALWAYS

    peak_waveform_max_s1_area = straxen.URLConfig(default=100, infer_type=False,
                                                  help='Maximum area of S1 we do hit analysis')
    hit_number_peak_waveform_s1 = straxen.URLConfig(default=10, type=int,
                                                    help='Number of hits we stored in peaks')
    hit_shadow_casting_pmt_pe_threshold = straxen.URLConfig(default=2, infer_type=False,
                                                            help='Minimum PMT areas we consider a shadow casting PMT')
    hit_shadow_casting_time_backward = straxen.URLConfig(default=1000e6, infer_type=False,
                                                         help='Maximum of searching time for a casting shadow hit')
    gain_model = straxen.URLConfig(infer_type=False,
                                   help='PMT gain model. Specify as (model_type, model_config)',
                                   )
    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    def infer_dtype(self):
        self.hits1_dtype_hit = [
            # name                dtype       comment
            ('hits_max_time', np.int64, 'Hits max_time'),
            ('hits_area', np.float64, 'Hits area'),
            ('hits_channel', np.int16, 'Hits channel'),
            ('hits_height', np.float64, 'Hits height'),
        ]
        self.hits1_dtype_shadow = [
            # name                dtype       comment
            ('hits_shadow_dt', np.int64, 'Hits time difference to prehits'),
            ('hits_shadow', np.float64, 'Hits shadow'),
            ('hits_prehits_area', np.float64, 'Hits prehits area'),
        ]
        self.hits1_dtype = self.hits1_dtype_shadow + self.hits1_dtype_hit
        dtype = strax.time_fields
        for item in self.hits1_dtype:
            (_name, _dtype, _comment) = item
            dtype = dtype + [
                ((f'{_comment} in peak level', f'peak_{_name}'), _dtype, (self.hit_number_peak_waveform_s1))]
        return dtype

    def setup(self):
        self.drift_time_max = int(straxen.tpc_z / self.electron_drift_velocity)
        return

    def get_window_size(self):
        return self.hit_shadow_casting_time_backward

    @staticmethod
    @numba.njit
    def hit_shadow(hits, pre_peaks,
                   touching_windows,
                   exponent,
                   pmt_area_threshold):
        '''
        Calculate hit shadow casted from a previous peak which has the largest
        area_in_pmt/dt
        '''
        # Loop hits
        for p_i, suspicious_hit in enumerate(hits):
            indices = touching_windows[p_i]
            # Loop peaks before certain hit
            for idx in range(indices[0], indices[1]):
                casting_peak = pre_peaks[idx]
                dt = suspicious_hit['hits_max_time'] - casting_peak['center_time']
                if dt <= 0:
                    continue
                # We only need the specific channel area for hit shadow
                casting_pmt_pe = casting_peak['area_per_channel'][suspicious_hit['hits_channel']]
                if casting_pmt_pe >= pmt_area_threshold:
                    new_shadow = casting_pmt_pe / dt
                    if new_shadow > hits['hits_shadow'][p_i]:
                        hits['hits_shadow'][p_i] = new_shadow
                        hits['hits_shadow_dt'][p_i] = dt
                        hits['hits_prehits_area'][p_i] = casting_pmt_pe

    def compute(self, peaks, hits):
        result = np.zeros(len(peaks), self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = peaks['endtime']
        result['peak_hits_channel'] = -1

        hitlet = np.zeros(len(hits), dtype=[(elem[0], elem[1]) for elem in self.hits1_dtype + strax.time_fields])
        for dtype in self.hits1_dtype_hit:
            hitlet[dtype[0]] = hits[dtype[0]]

        # Searching window is from hit_shadow_casting_time_backward to hits_max_time
        roi_shadow = np.zeros(len(hitlet), dtype=strax.time_fields)
        roi_shadow['time'] = hitlet['hits_max_time'] - self.hit_shadow_casting_time_backward
        roi_shadow['endtime'] = hitlet['hits_max_time']
        # Use temp_peaks endtime as time to search, to avoid the hits' own peak to join the shadow
        temp_peaks = np.zeros(len(peaks), dtype=strax.time_fields)
        temp_peaks['time'] = temp_peaks['endtime'] = peaks['endtime']
        split_shadow_window = strax.touching_windows(temp_peaks, roi_shadow)
        self.hit_shadow(hits=hitlet,
                        pre_peaks=peaks,
                        touching_windows=split_shadow_window,
                        exponent=int(-1),
                        pmt_area_threshold=self.hit_shadow_casting_pmt_pe_threshold)

        # Perform waveform analysis only on records contained in small S1s
        mask = (peaks['type'] == 1) & (peaks['area'] <= self.peak_waveform_max_s1_area)
        # Search for hit-s1 matching with maxtime
        temp = np.zeros(len(hitlet), dtype=strax.time_fields)
        temp['time'] = temp['endtime'] = hitlet['hits_max_time']
        split_hits_window = strax.touching_windows(temp, peaks[mask])
        for i, index in zip(range(mask.sum()),
                            np.arange(len(mask))[mask]):
            left_i, right_i = split_hits_window[i]
            hitlet_head = hitlet[left_i:right_i][:self.hit_number_peak_waveform_s1]
            for name in [elem[0] for elem in self.hits1_dtype]:
                if 'channel' in name:
                    constant_values = -1
                else:
                    constant_values = 0
                result[index][f'peak_{name}'] = np.pad(hitlet_head[name],
                                                       (0, self.hit_number_peak_waveform_s1 - len(hitlet_head)),
                                                       constant_values=constant_values)
        return result


class PeakAmbienceS1(strax.OverlapWindowPlugin):
    """
    Return the first peak_number lone_hits information in a window near S1
    """
    __version__ = "0.0.0"
    parallel = True
    depends_on = ('peak_basics', 'lone_hits',)
    provides = 'peak_ambience_s1'
    data_kind = 'peaks'
    save_when = strax.SaveWhen.ALWAYS

    hit_number_peak_ambience_s1 = straxen.URLConfig(default=20, type=int,
                                                    help='Number of hits we stored in peaks for ambience')
    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (('Lone hits time in peak level', 'lone_hits_time'), np.int64, (self.hit_number_peak_ambience_s1)),
            (('Lone hits area in peak level', 'lone_hits_area'), np.float64, (self.hit_number_peak_ambience_s1)),
            (('Lone hits channel in peak level', 'lone_hits_channel'), np.int16, (self.hit_number_peak_ambience_s1))]
        return dtype

    def setup(self):
        self.drift_time_max = int(straxen.tpc_z / self.electron_drift_velocity)
        self.s1_ambience_time_window = 5 * self.drift_time_max
        return

    def get_window_size(self):
        return 10 * self.drift_time_max

    @staticmethod
    @numba.njit
    def fill_result(result, _dtype, windows, container, things_time_only,
                    things_dtype_only, peak_number):
        for i, _container in enumerate(container):
            left_i, right_i = windows[i]
            _split_things_time = things_time_only[left_i:right_i]
            _split_things_dtype = things_dtype_only[left_i:right_i]
            time_to_container = np.abs(_container['time'] - _split_things_time)
            _split_things_dtype = _split_things_dtype[np.argsort(time_to_container)][:peak_number]
            if 'channel' in _dtype:
                fill_in = np.ones(peak_number) * (-1)
            else:
                fill_in = np.zeros(peak_number)
            length = np.int8(len(_split_things_dtype))
            fill_in[:length] = _split_things_dtype
            result[i] = fill_in

    def compute(self, peaks, lone_hits):
        windows = strax.touching_windows(lone_hits, peaks,
                                         window=self.s1_ambience_time_window)
        result = np.zeros(len(peaks), self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = peaks['endtime']
        dtype = ['time', 'area', 'channel']
        for _dtype in dtype:
            self.fill_result(result=result[f'lone_hits_{_dtype}'],
                             _dtype=_dtype,
                             windows=windows,
                             container=peaks,
                             things_time_only=lone_hits['time'],
                             things_dtype_only=lone_hits[_dtype],
                             peak_number=self.hit_number_peak_ambience_s1)
        return result
