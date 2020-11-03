import strax
import numpy as np
import numexpr

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        'area_vs_width_nbins',
        type=int, default=60,
        help='Number of bins for area vs width histogram for online monitor. '
             'NB: this is a 2D histogram'),
    strax.Option(
        'area_vs_width_bounds',
        type=tuple, default=((0, 5), (0, 5)),
        help='Boundaries of log-log histogram of area vs width'),
    strax.Option(
        'area_vs_width_min_gap',
        type=int, default=20,
        help='Minimal gap between consecutive peaks to be considered for the '
             '"area_vs_width_hist_clean" To turn off this cut, set to 0.'),
    strax.Option(
        'area_vs_width_cut_string',
        type=str, default='',
        help='Selection (like selection_str) applied to data for '
             '"area_vs_width_hist_clean", cuts should be separated using "&"'
             'For example: (tight_coincidence > 2) & (area_fraction_top < 0.1)'
             'Default is no selection (other than "area_vs_width_min_gap")'),
    strax.Option(
            'lone_hits_area_bounds',
            type=tuple, default=(0, 1500),
            help='Boundaries area histogram of lone hits [ADC]'),
    strax.Option(
            'lone_hits_area_nbins',
            type=int, default=100,
            help='Number of bins of histogram of lone_hits_area for online '
                 'monitor'),
    strax.Option(
        'n_tpc_pmts', type=int,
        help='Number of TPC PMTs'),
)
class OnlinePeakMonitor(strax.Plugin):
    """
    Plugin to write data to the online-monitor. Data that is written by
    this plugin should be small such as to not overload the runs-
    database.

    This plugin takes 'peaks_basics' and 'lone_hits'. Although they are
    not strictly related, they are aggregated into a single data_type
    in order to minimize the number of documents in hte online monitor.

    Produces 'online_peak_monitor' with info on the lone-hits and peaks
    """
    depends_on = ('peak_basics', 'lone_hits')
    provides = 'online_peak_monitor'
    __version__ = '0.0.2'

    def infer_dtype(self):
        n_bins_area_width = self.config['area_vs_width_nbins']
        bounds_area_width = self.config['area_vs_width_bounds']

        bounds_lone_hit_area = self.config['lone_hits_area_bounds']
        n_bins_lone_hit_area = self.config['lone_hits_area_nbins']

        n_tpc_pmts = self.config['n_tpc_pmts']
        dtype = [
            (('Start time of the chunk', 'time'),
             np.int64),
            (('End time of the chunk', 'endtime'),
             np.int64),
            (('Area vs width histogram (log-log)', 'area_vs_width_hist'),
             (np.int64, (n_bins_area_width, n_bins_area_width))),
            (('Area vs width edges (log-space)', 'area_vs_width_bounds'),
             (np.float64, np.shape(bounds_area_width))),
            (('Area vs width histogram with cuts (log-log)', 'area_vs_width_hist_clean'),
             (np.int64, (n_bins_area_width, n_bins_area_width))),
            (('Lone hits areas histogram [ADC-counts]', 'lone_hits_area_hist'),
             (np.int64, n_bins_lone_hit_area)),
            (('Lone hits areas bounds [ADC-counts]', 'lone_hits_area_bounds'),
             (np.float64, np.shape(bounds_lone_hit_area))),
            (('Lone hits per channel', 'lone_hits_per_channel'),
             (np.int64, n_tpc_pmts)),
        ]
        return dtype

    def compute(self, peaks, lone_hits, start, end):
        # General setup
        res = np.zeros(1, dtype=self.dtype)
        res['time'] = start
        res['endtime'] = end

        # Bounds for histograms
        res['area_vs_width_bounds'] = self.config['area_vs_width_bounds']
        res['lone_hits_area_bounds'] = self.config['lone_hits_area_bounds']

        # -- Peak vs area 2D histogram --
        # Always cut out unphysical peaks
        sel = (peaks['area'] > 0) & (peaks['range_50p_area'] > 0)

        hist, _, _ = np.histogram2d(
            np.log10(peaks[sel]['area']),
            np.log10(peaks[sel]['range_50p_area']),
            range=self.config['area_vs_width_bounds'],
            bins=self.config['area_vs_width_nbins'])
        res['area_vs_width_hist'] = hist.T

        # Experimental example of how to apply cuts here.
        # Let's make a cut on the time between two peaks, if too short,
        # ignore the peak.
        timedelta = peaks[1:]['time'] - strax.endtime(peaks)[:-1]
        timesel = timedelta > self.config['area_vs_width_min_gap']
        # Last peak always has no tails
        timesel = np.concatenate((timesel, [True]))
        sel &= timesel

        # TODO
        #  somehow prevent overlap with strax.context.apply_selection
        # Also apply the area_vs_width_cut_string like a selection_str
        selection_string = self.config['area_vs_width_cut_string']
        if selection_string != '':
            if isinstance(selection_string, (list, tuple)):
                selection_string = ' & '.join(f'({x})' for x in selection_string)

            mask = numexpr.evaluate(selection_string, local_dict={
                fn: peaks[fn]
                for fn in peaks.dtype.names})
            sel &= mask

        clean_hist, _, _ = np.histogram2d(
            np.log10(peaks[sel]['area']),
            np.log10(peaks[sel]['range_50p_area']),
            range=self.config['area_vs_width_bounds'],
            bins=self.config['area_vs_width_nbins'])
        res['area_vs_width_hist_clean'] = clean_hist.T

        # -- Lone hit properties --
        # Make histogram of ADC counts
        lone_hit_areas, _ = np.histogram(
            lone_hits['area'],
            bins=self.config['lone_hits_area_nbins'],
            range=self.config['lone_hits_area_bounds'])

        # Count number of lone-hits per PMT
        lone_hit_channel_count, _ = np.histogram(
            lone_hits['channel'],
            bins=self.config['n_tpc_pmts'],
            range=[0, self.config['n_tpc_pmts']])
        res['lone_hits_area_hist'] = lone_hit_areas
        res['lone_hits_per_channel'] = lone_hit_channel_count

        # Cleanup
        del hist, clean_hist, lone_hit_areas, lone_hit_channel_count
        return res
