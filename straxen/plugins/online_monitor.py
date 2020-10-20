import strax
import numpy as np

export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option(
        'area_vs_width_nbins',
        type=int, default=100,
        help='Number of bins for online monitor to aggreage'),
    strax.Option(
        'area_vs_width_bounds',
        type=tuple, default=((0, 6), (0, 6)),
        help='Boundaries of log-log histogram of area vs width'),
    strax.Option(
        'area_vs_width_min_gap',
        type=int, default=20,
        help='Minimal gap between consecutive peaks to be considered for the '
             '"area_vs_width_hist_clean" To turn off this cut, set to 0.')
)
class OnlinePeakMonitor(strax.Plugin):
    depends_on = ('peak_basics',)
    provides = 'online_peak_monitor'
    __version__ = '0.0.0'

    def infer_dtype(self):
        nbins = self.config['area_vs_width_nbins']
        bounds = self.config['area_vs_width_bounds']
        dtype = [
            (('Start time of the chunk', 'time'), np.int64),
            (('End time of the chunk', 'endtime'), np.int64),
            (('Area vs width histogram', 'area_vs_width_hist'), (np.int64, (nbins, nbins))),
            (('Area vs width edges', 'area_vs_width_bounds'), (np.float64, np.shape(bounds))),
            (
                ('Experimental, Area vs width histogram with cuts',
                 'area_vs_width_hist_clean'),
                (np.int64, (nbins, nbins))),
        ]
        return dtype

    def compute(self, peaks, start, end):
        res = np.zeros(1, dtype=self.dtype)
        res['time'] = start
        res['endtime'] = end
        res['area_vs_width_bounds'] = self.config['area_vs_width_bounds']
        sel = (peaks['area'] > 0) & (peaks['range_50p_area'] > 0)

        hist, _, _ = np.histogram2d(
            np.log10(peaks[sel]['area']),
            np.log10(peaks[sel]['range_50p_area']),
            range=self.config['area_vs_width_bounds'],
            bins=self.config['area_vs_width_nbins'])
        res['area_vs_width_hist'] = hist.T

        # Experimental example
        # Let's make a cut on the time between two peaks, if too short, ignore
        # the peak
        timedelta = peaks[1:]['time'] - strax.endtime(peaks)[:-1]
        timesel = timedelta > self.config['area_vs_width_min_gap']
        # Last peak always has no tails
        timesel = np.concatenate((timesel, [True]))
        sel &= timesel

        clean_hist, _, _ = np.histogram2d(
            np.log10(peaks[sel]['area']),
            np.log10(peaks[sel]['range_50p_area']),
            range=self.config['area_vs_width_bounds'],
            bins=self.config['area_vs_width_nbins'])
        res['area_vs_width_hist_clean'] = clean_hist.T

        del hist, clean_hist, _, _
        return res
