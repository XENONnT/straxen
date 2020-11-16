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
        'online_monitor_nbins',
        type=int, default=100,
        help='Number of bins of histogram of online monitor. Will be used '
             'for: '
             'lone_hits_area-histogram, '
             'area_fraction_top-histogram, '
             'near_s1_hists, '),
    strax.Option(
        'near_s1_hists_cut_string',
        type=str,
        default='(n_channels > 20) & (n_channels < 400) & (area < 1000) & '
                '(area > 5) & (rise_time < 100) & (type == 1)',
        help='Selection (like selection_str) applied to data for '
             '"near_s1_hists", cuts should be separated using "&"'
             'For example: (tight_coincidence > 2) & (area_fraction_top < 0.1)'
             'Default is no selection (other than "area_vs_width_min_gap")'),
    strax.Option(
        'near_s1_hists_bounds',
        type=tuple,
        default=(0, 1000),
        help='Bounds for the near s1-peaks in PE'),
    strax.Option(
        'near_s1_max_time_diff',
        type=int, default=2_000,
        help='Max gap between two peaks for the near-s1 area histogram [ns]'),
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
    __version__ = '0.0.4'
    # TODO make new datakind:
    # data_kind = 'online_monitor'
    rechunk_on_save = False

    def infer_dtype(self):
        n_bins_area_width = self.config['area_vs_width_nbins']
        bounds_area_width = self.config['area_vs_width_bounds']

        n_bins = self.config['online_monitor_nbins']

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
             (np.int64, n_bins)),
            (('Lone hits areas bounds [ADC-counts]', 'lone_hits_area_bounds'),
             (np.float64, 2)),
            (('Lone hits per channel', 'lone_hits_per_channel'),
             (np.int64, n_tpc_pmts)),
            (('AFT histogram', 'aft_hist'),
             (np.int64, n_bins)),
            (('AFT bounds', 'aft_bounds'),
             (np.float64, 2)),
            (('Number of contributing channels histogram', 'n_channel_hist'),
             (np.int64, n_tpc_pmts)),
            (('Number of contributing channels histogram bounds', 'n_channel_bounds'),
             (np.float64, 2)),
            (('Near S1 peaks area hist', 'near_s1_area_hist'),
             (np.int64, n_bins)),
            (('Near S1 peaks area hist bounds', 'near_s1_area_bounds'),
             (np.float64, 2)),
        ]
        return dtype

    def compute(self, peaks, lone_hits, start, end):
        # General setup
        res = np.zeros(1, dtype=self.dtype)
        res['time'] = start
        res['endtime'] = end
        n_pmt = self.config['n_tpc_pmts']
        n_bins = self.config['online_monitor_nbins']

        # Bounds for histograms
        res['area_vs_width_bounds'] = self.config['area_vs_width_bounds']
        res['lone_hits_area_bounds'] = self.config['lone_hits_area_bounds']

        # -- Peak vs area 2D histogram --
        # Always cut out unphysical peaks
        sel = (peaks['area'] > 0) & (peaks['range_50p_area'] > 0)
        res['area_vs_width_hist'] = self.area_width_hist(peaks[sel])

        # Experimental example of how to apply cuts here.
        # Let's make a cut on the time between two peaks, if too short,
        # ignore the peak.
        timedelta = peaks[1:]['time'] - strax.endtime(peaks)[:-1]
        timesel = timedelta > self.config['area_vs_width_min_gap']
        # Last peak always has no tails
        timesel = np.concatenate((timesel, [True]))
        sel &= timesel

        # Also apply the area_vs_width_cut_string like a selection_str
        sel_str = self.config['area_vs_width_cut_string']
        sel = self._config_as_selection_str(sel_str, peaks, pre_sel=sel)
        res['area_vs_width_hist_clean'] = self.area_width_hist(peaks[sel])
        # make a new selection don't re-use
        del sel

        # -- Lone hit properties --
        # Make histogram of ADC counts
        lone_hit_areas, _ = np.histogram(lone_hits['area'],
                                         bins=n_bins,
                                         range=self.config['lone_hits_area_bounds'])

        lone_hit_channel_count, _ = np.histogram(lone_hits['channel'],
                                                 bins=n_pmt,
                                                 range=[0, n_pmt])
        # Count number of lone-hits per PMT
        res['lone_hits_area_hist'] = lone_hit_areas
        res['lone_hits_per_channel'] = lone_hit_channel_count

        # -- AFT histogram --
        aft_b = [0, 1]
        aft_hist, _ = np.histogram(peaks['area_fraction_top'], bins=n_bins, range=aft_b)
        res['aft_hist'] = aft_hist
        res['aft_bounds'] = aft_b

        # -- Number of contributing channels channels --
        n_cont_b = [0, n_pmt]
        n_cont_hist, _ = np.histogram(peaks['n_channels'], bins=n_pmt, range=n_cont_b)
        res['n_channel_hist'] = n_cont_hist
        res['n_channel_bounds'] = n_cont_b

        # -- Experimental selection --
        # We first apply a basic selection on the peaks to e.g. get S1s
        mask = self._config_as_selection_str(self.config['near_s1_hists_cut_string'], peaks)
        peaks_sel = peaks[mask]
        # We select peaks where the peak before or the peak after it is within
        # near_s1_max_time_diff ns. TODO: Do we want another hist for this?
        time_diff = peaks_sel[1:]['time'] - strax.endtime(peaks_sel)[:-1]
        is_close = time_diff < self.config['near_s1_max_time_diff']
        time_mask = np.zeros(len(peaks_sel), dtype=np.bool_)
        # Either the previous or the next peak can be close, take both into account
        time_mask[:-1] = is_close
        time_mask[1:] = time_mask[1:] | is_close

        # Make the area hist
        near_s1_bound = self.config['near_s1_hists_bounds']
        near_s1_hist, _ = np.histogram(peaks_sel['area'][time_mask], bins=n_bins, range=near_s1_bound)
        res['near_s1_area_hist'] = near_s1_hist
        res['near_s1_area_bounds'] = near_s1_bound

        # Cleanup
        # del hist, clean_hist, lone_hit_areas, lone_hit_channel_count
        return res

    # TODO
    #  somehow prevent overlap with strax.context.apply_selection
    @staticmethod
    def _config_as_selection_str(selection_string, data, pre_sel=None):
        """Get mask for data base on the selection string"""
        if pre_sel is None:
            pre_sel = np.ones(len(data), dtype=np.bool_)

        if selection_string != '':
            if isinstance(selection_string, (list, tuple)):
                selection_string = ' & '.join(f'({x})' for x in selection_string)

            mask = numexpr.evaluate(selection_string, local_dict={
                fn: data[fn]
                for fn in data.dtype.names})
            pre_sel &= mask
        return pre_sel

    def area_width_hist(self, data):
        """Make area vs width 2D-hist"""
        hist, _, _ = np.histogram2d(
            np.log10(data['area']),
            np.log10(data['range_50p_area']),
            range=self.config['area_vs_width_bounds'],
            bins=self.config['area_vs_width_nbins'])
        return hist.T


class OnlineMonitor(strax.LoopPlugin):
    """
    Loop over the online-monitor chunks, get the veto intervals that are within
    each of these chunks. Compute the live-time within each of the chunks.
    """
    depends_on = ('online_peak_monitor', 'veto_intervals')
    provides = 'online_monitor'
    __version__ = '0.0.4'
    rechunk_on_save = False

    def infer_dtype(self):
        dtype = strax.unpack_dtype(self.deps['online_peak_monitor'].dtype_for('online_peak_monitor'))
        dtype += [(('Live time', 'live_time'),
                   np.float64),]
        return dtype

    def compute_loop(self, peaks, veto_intervals):
        res = {}
        for d in peaks.dtype.names:
            res[d] = peaks[d]
        dt = strax.endtime(peaks) - peaks['time']
        assert not np.iterable(dt) or len(dt) == 1
        if dt > 0:
            res['live_time'] = 1 - np.sum(veto_intervals['veto_interval'])/dt
        else:
            res['live_time'] = 1
        return res
