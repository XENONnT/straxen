import numpy as np
import numba
import strax
import straxen
from straxen.get_corrections import get_correction_from_cmt

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(name='pre_s2_area_threshold', default=1000,
                 help='Only take S2s large than this into account'
                 ' when calculating PeakShadow [PE]'),
    strax.Option(name='deltatime_exponent', default=-1.0,
                 help='The exponent of delta t when calculating shadow'),
    strax.Option('time_window_backward', default=int(3e9),
                 help='Search for S2s causing shadow in this time window [ns]'),
    strax.Option(name='electron_drift_velocity',
                 default=('electron_drift_velocity', 'ONLINE', True),
                 help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'),
    strax.Option(name='max_drift_length', default=straxen.tpc_z,
                 help='Total length of the TPC from the bottom of gate to the'
                 ' top of cathode wires [cm]'), 
    strax.Option(name='exclude_drift_time', default=False,
                 help=' Whether deduct max drift time to avoid peak interference in a single event [ns]'))
class PeakShadow(strax.OverlapWindowPlugin):
    '''
    This plugin can find and calculate the previous S2 shadow at peak level,
    with time window backward and previous S2 area as options.
    It also gives the area and position infomation of these previous S2s.
    '''

    __version__ = '0.0.3'
    depends_on = ('peak_basics', 'peak_positions')
    provides = 'peak_shadow'
    save_when = strax.SaveWhen.EXPLICIT

    def setup(self):
        self.n_seconds = self.config['time_window_backward']
        if self.config['exclude_drift_time']:
            electron_drift_velocity = get_correction_from_cmt(
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
                 ('shadow_dt', np.int64, 'time diffrence to the previous s2 [ns]'),
                 ('pre_s2_x', np.float32, 'x of previous s2 peak causing shadow [cm]'),
                 ('pre_s2_y', np.float32, 'y of previous s2 peak causing shadow [cm]')]
        dtype += strax.time_fields
        return dtype

    def compute(self, peaks):
        roi_shadow = np.zeros(len(peaks), dtype=strax.time_fields)
        roi_shadow['time'] = peaks['center_time'] - self.n_seconds
        roi_shadow['endtime'] = peaks['center_time'] - self.n_drift_time

        mask_pre_s2 = peaks['area'] > self.s2_threshold
        mask_pre_s2 &= peaks['type'] == 2
        split_peaks = strax.split_touching_windows(peaks[mask_pre_s2], roi_shadow)
        res = np.zeros(len(peaks), self.dtype)
        res['pre_s2_x'] = np.nan
        res['pre_s2_y'] = np.nan
        compute_shadow(peaks, split_peaks, self.exponent, res)

        res['time'] = peaks['time']
        res['endtime'] = strax.endtime(peaks)

        return res


def compute_shadow(peaks, split_peaks, exponent, res):
    '''
    The function to compute shadow of each peak
    '''
    if len(res):
        return _compute_shadow(peaks, split_peaks, exponent, res)


@numba.njit(cache=True)
def _compute_shadow(peaks, split_peaks, exponent, res):
    '''
    Numba accelerated shadow calculation
    '''
    for p_i, p_a in enumerate(peaks):
        new_shadow = 0
        for s2_a in split_peaks[p_i]:
            if p_a['center_time'] - s2_a['center_time'] <= 0:
                continue
            new_shadow = s2_a['area'] * (p_a['center_time'] - s2_a['center_time'])**exponent
            if new_shadow > res['shadow'][p_i]:
                res['shadow'][p_i] = new_shadow
                res['pre_s2_area'][p_i] = s2_a['area']
                res['shadow_dt'][p_i] = p_a['center_time'] - s2_a['center_time']
                res['pre_s2_x'][p_i] = s2_a['x']
                res['pre_s2_y'][p_i] = s2_a['y']
