from straxen.plugins.peak_processing import PeakBasics
import straxen
import strax
import numpy as np
import numba
export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('hit_number_peak_waveform_s1', default=10, type=int,
                 help='Number of hits we stored in peaks'),
)
class EventWaveformS1(strax.Plugin):
    """
    Trivially fill peak_waveform_s1 into event level
    """
    __version__ = '0.0.0'
    depends_on = ('event_basics', 'peak_basics', 'peak_waveform_s1')
    provides = 'event_waveform_s1'

    def setup(self):
        # The fill-in dtype needs to be consistent with self.infer_dtype
        return

    def infer_dtype(self):
        self.hits1_dtype = [
                            # name                dtype       comment
                            ('hits_max_time',     np.int64,   'Hits max_time'),
                            ('hits_area',         np.float64, 'Hits area'),
                            ('hits_channel',      np.int16,   'Hits channel'),
                            ('hits_height',       np.float64, 'Hits height'),
                            ]
        self.hits1_dtype += [
                            # name                dtype       comment
                            ('hits_shadow_dt',     np.int64,   'Hits time difference to prehits'),
                            ('hits_shadow',        np.float64, 'Hits shadow'),
                            ('hits_prehits_area',  np.float64, 'Hits prehits area'),
                            ]
        dtype = strax.time_fields
        for item in self.hits1_dtype:
            (_name, _dtype, _comment) = item
            dtype = dtype + [((f'{_comment} in event level', f's1_{_name}'), _dtype, (self.config['hit_number_peak_waveform_s1']))]
        return dtype

    def compute(self, events, peaks):
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        split_peaks = strax.split_by_containment(peaks, result)

        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            idx = event['s1_index']
            if idx >= 0:
                for dtype in self.hits1_dtype:
                    result[f's1_{dtype[0]}'][event_i] = sp[f'peak_{dtype[0]}'][idx]
        return result


@export
@strax.takes_config(
    strax.Option('hit_number_peak_ambience_s1', default=20, type=int,
                 help='Number of hits we stored in peaks for ambience'),
)
class EventAmbienceS1(strax.Plugin):
    """
    Trivially fill peak_ambience_s1 into event level
    """
    __version__ = '0.0.0'
    depends_on = ('event_basics', 'peak_basics', 'peak_ambience_s1')
    provides = 'event_ambience_s1'

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (('Lone hits time in event level', 's1_ambience_lone_hits_time'), np.int64, (self.config['hit_number_peak_ambience_s1'])),
            (('Lone hits area in event level', 's1_ambience_lone_hits_area'), np.float64, (self.config['hit_number_peak_ambience_s1'])),
            (('Lone hits channel in event level', 's1_ambience_lone_hits_channel'), np.int16, (self.config['hit_number_peak_ambience_s1']))]
        return dtype

    def compute(self, events, peaks):
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        split_peaks = strax.split_by_containment(peaks, result)

        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            idx = event['s1_index']
            if idx >= 0:
                for dtype in ['time', 'area', 'channel']:
                    result[f's1_ambience_lone_hits_{dtype}'][event_i] = sp[f'lone_hits_{dtype}'][idx]
        return result



@export
@strax.takes_config(
    strax.Option('peaklet_number_event_waveform_s2', default=20, type=int,
                 help='Number of peaklet we store'),
)
class EventWaveformS2(strax.OverlapWindowPlugin):
    """
    Return timing and area for single peaklets in S2s, in small S2s they are single electrons.
    """
    __version__ = "0.0.0"
    parallel = True
    depends_on = ('event_info', 'peaklets',)
    provides = 'event_waveform_s2'
    save_when = strax.SaveWhen.ALWAYS

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    def setup(self):
        self.drift_time_max = int(straxen.tpc_z / self.electron_drift_velocity)
        return

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (('Center time of the peaklets in S2', 's2_peaklet_center_time'), np.int64, (self.config['peaklet_number_event_waveform_s2'])),
            (('Area of the peaklets in S2', 's2_peaklet_area'), np.float32, (self.config['peaklet_number_event_waveform_s2'])),
        ]
        return dtype

    def get_window_size(self):
        return 10 * self.drift_time_max

    def compute(self, events, peaklets):
        # Step1: Split peaklets contained in main S2 time and endtime
        temp_events = np.zeros(len(events), dtype=strax.time_fields)
        temp_events['time'] = events['s2_time']
        temp_events['endtime'] = events['s2_endtime']
        split_peaklets = strax.split_touching_windows(peaklets, temp_events)

        # Step 2: Fetch the first peaklet_number peaklets, and store area and center_time
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        for i in range(len(split_peaklets)):
            peaklets_head = split_peaklets[i][:self.config['peaklet_number_event_waveform_s2']]
            result[i]['s2_peaklet_area'] = np.pad(peaklets_head['area'], (0, self.config['peaklet_number_event_waveform_s2'] - len(peaklets_head)))
            result[i]['s2_peaklet_center_time'] = np.pad(
                                                peaklets_head['time'] + PeakBasics.compute_center_times(peaklets_head),
                                                (0, self.config['peaklet_number_event_waveform_s2'] - len(peaklets_head)))
        return result


@export
@strax.takes_config(
    strax.Option('peak_number_event_ambience_s2', default=30, type=int,
                 help='Number of peaks we store near the main S2 ambience'),
)
class EventAmbienceS2(strax.OverlapWindowPlugin):
    """
    Return the first peak_number peaks' information, sorted according to its time difference to main_S2.
    """
    __version__ = "0.0.0"
    parallel = True
    depends_on = ('event_basics', 'peak_basics',)
    provides = 'event_ambience_s2'
    save_when = strax.SaveWhen.ALWAYS

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    def setup(self):
        self.drift_time_max = int(straxen.tpc_z / self.electron_drift_velocity)
        return

    def infer_dtype(self):
        peak_number = self.config['peak_number_event_ambience_s2']
        dtype = strax.time_fields + [
            (('S2 Ambience Peak range_50p_area in event level', 's2_ambience_peak_range_50p_area'), np.int64, (peak_number)),
            (('S2 Ambience Peak range_90p_area in event level', 's2_ambience_peak_range_90p_area'), np.float64, (peak_number)),
            (('S2 Ambience Peak area_fraction_top in event level', 's2_ambience_peak_area_fraction_top'), np.float64, (peak_number)),
            (('S2 Ambience Peak center_time in event level', 's2_ambience_peak_center_time'), np.float64, (peak_number)),
            (('S2 Ambience Peak max_pmt_area in event level', 's2_ambience_peak_max_pmt_area'), np.float64, (peak_number)),
            (('S2 Ambience Peak area in event level', 's2_ambience_peak_area'), np.float64, (peak_number)),
            (('S2 Ambience Peak type in event level', 's2_ambience_peak_type'), np.int, (peak_number)),
        ]
        return dtype

    def get_window_size(self):
        return 10 * self.drift_time_max

    def fill_result(self, result, split_peaks, events, dtype):
        for i, (_split_peaks, _events) in enumerate(zip(split_peaks, events)):
            time_to_main_s2 = np.abs(_events['s2_time'] - _split_peaks['time'])
            _split_peaks = _split_peaks[np.argsort(time_to_main_s2)]
            peaks_head = _split_peaks[:self.config['peak_number_event_ambience_s2']]
            for name in dtype:
                if name == 'type':
                    constant_values = -1
                else:
                    constant_values = 0
                result[i][f's2_ambience_peak_{name}'] = np.pad(peaks_head[name],
                                                               (0, self.config['peak_number_event_ambience_s2'] - len(peaks_head)),
                                                               constant_values=constant_values)
        return result

    def compute(self, events, peaks):
        # Step1: Find peaks contained in the drift_time_max window before and after
        temp_events = np.zeros(len(events), dtype=strax.time_fields)
        temp_events['time'] = events['s2_time'] - self.drift_time_max
        temp_events['endtime'] = events['s2_endtime'] + self.drift_time_max
        split_peaks = strax.split_by_containment(peaks, temp_events)

        # Step2: Fill the peak level information in
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        dtype = ['range_50p_area', 'range_90p_area', 'center_time',
                 'area_fraction_top', 'max_pmt_area', 'area', 'type']
        result = self.fill_result(result, split_peaks, events, dtype)
        return result


