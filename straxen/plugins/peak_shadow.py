import numpy as np
import strax
import numba
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('pre_s2_area_threshold', default=1000,
                 help='Only take S2s large than this into account when calculating PeakShadow.'),
    strax.Option('time_window_backward', default=int(3e9),
                 help='Search for S2s causing shadow in this time window'))
class PeakShadow(strax.Plugin):
    """
    This plugin can find and calculate the previous S2 shadow at peak level,
    with time window backward and previous S2 area as options.
    It also gives the area and position infomation of these previous S2s.
    """
    __version__ = '0.0.1'
    depends_on = ('peak_basics', 'peak_positions')
    provides = 'peak_shadow'
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        dtype = [('pre_s2_area', np.float32, 'previous s2 area [PE]'),
                 ('shadow_dt', np.int64, 'time diffrence to the previous s2 [ns]'),
                 ('shadow', np.float32, 'previous s2 shadow [PE/ns]')]
        dtype += strax.time_fields
        return dtype

    def compute(self, peaks):
        roi_dt = np.dtype([(('when s1 ends', 'time'), int), (('till the upper limit of drfit time', 'endtime'), int)])
        roi_shadow = np.zeros(len(peaks), dtype=roi_dt)
        n_seconds = self.config['time_window_backward']
        roi_shadow['time'] = peaks['time'] - n_seconds
        roi_shadow['endtime'] = peaks['time']

        mask_pre_s2 = (peaks['area'] > self.config['pre_s2_area_threshold']) & (peaks['type']==2)
        split_peaks = strax.split_touching_windows(peaks[mask_pre_s2], roi_shadow)
        res = np.zeros(len(peaks), self.dtype)
        compute_shadow(peaks, split_peaks, res)

        res['time'] = peaks['time']
        res['endtime'] = strax.endtime(peaks)

        return res

def compute_shadow(peaks, split_peaks, res):
    if len(res):
        return _compute_shadow(peaks, split_peaks, res)

@numba.njit()
def _compute_shadow(peaks, split_peaks, res):
    for p_i, p_a in enumerate(peaks):
        new_shadow = 0
        for s2_a in split_peaks[p_i]:
            new_shadow = s2_a['area'] / (p_a['time'] - s2_a['center_time'])
            if new_shadow > res['shadow'][p_i]:
                res['pre_s2_area'][p_i] = s2_a['area']
                res['shadow'][p_i] = new_shadow
                res['shadow_dt'][p_i] = p_a['time'] - s2_a['center_time']
