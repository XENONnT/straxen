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
        'online_peak_monitor_nbins',
        type=int, default=100,
        help='Number of bins of histogram of online monitor. Will be used '
             'for: '
             'lone_hits_area-histogram, '
             'area_fraction_top-histogram, '
             'online_se_gain estimate (histogram is not stored), '
    ),
    strax.Option(
        'lone_hits_cut_string',
        type=str,
        default='(area >= 50) & (area <= 250)',
        help='Selection (like selection_str) applied to data for '
             '"lone-hits", cuts should be separated using "&")'),
    strax.Option(
        'lone_hits_min_gap',
        type=int,
        default=15_000,
        help='Minimal gap [ns] between consecutive lone-hits. To turn off '
             'this cut, set to 0.'),
    strax.Option(
        'n_tpc_pmts', type=int,
        help='Number of TPC PMTs'),
    strax.Option(
        'online_se_bounds',
        type=tuple, default=(7, 70),
        help='Window for online monitor [PE] to look for the SE gain, value'
    )
)
class OnlinePeakMonitor(strax.Plugin):
    """
    Plugin to write data to the online-monitor. Data that is written by
    this plugin should be small such as to not overload the runs-
    database.

    This plugin takes 'peak_basics' and 'lone_hits'. Although they are
    not strictly related, they are aggregated into a single data_type
    in order to minimize the number of documents in the online monitor.

    Produces 'online_peak_monitor' with info on the lone-hits and peaks
    """
    depends_on = ('peak_basics', 'lone_hits')
    provides = 'online_peak_monitor'
    data_kind = 'online_peak_monitor'
    __version__ = '0.0.5'
    rechunk_on_save = False

    def infer_dtype(self):
        n_bins_area_width = self.config['area_vs_width_nbins']
        bounds_area_width = self.config['area_vs_width_bounds']

        n_bins = self.config['online_peak_monitor_nbins']

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
            (('Single electron gain', 'online_se_gain'),
             np.float32),
        ]
        return dtype

    def compute(self, peaks, lone_hits, start, end):
        # General setup
        res = np.zeros(1, dtype=self.dtype)
        res['time'] = start
        res['endtime'] = end
        n_pmt = self.config['n_tpc_pmts']
        n_bins = self.config['online_peak_monitor_nbins']

        # Bounds for histograms
        res['area_vs_width_bounds'] = self.config['area_vs_width_bounds']
        res['lone_hits_area_bounds'] = self.config['lone_hits_area_bounds']

        # -- Peak vs area 2D histogram --
        # Always cut out unphysical peaks
        sel = (peaks['area'] > 0) & (peaks['range_50p_area'] > 0)
        res['area_vs_width_hist'] = self.area_width_hist(peaks[sel])
        del sel

        # -- Lone hit properties --
        # Make a mask with the cuts.
        # NB: LONE HITS AREA ARE IN ADC!
        mask = _config_as_selection_str(
            self.config['lone_hits_cut_string'], lone_hits)
        # Now only take lone hits that are separated in time.
        lh_timedelta = lone_hits[1:]['time'] - strax.endtime(lone_hits)[:-1]
        # Hits on the left are far away? (assume first is because of chunk bound)
        mask &= np.hstack([True, lh_timedelta > self.config['lone_hits_min_gap']])
        # Hits on the right are far away? (assume last is because of chunk bound)
        mask &= np.hstack([lh_timedelta > self.config['lone_hits_min_gap'], True])

        # Make histogram of ADC counts
        lone_hit_areas, _ = np.histogram(lone_hits[mask]['area'],
                                         bins=n_bins,
                                         range=self.config['lone_hits_area_bounds'])

        lone_hit_channel_count, _ = np.histogram(lone_hits[mask]['channel'],
                                                 bins=n_pmt,
                                                 range=[0, n_pmt])
        # Count number of lone-hits per PMT
        res['lone_hits_area_hist'] = lone_hit_areas
        res['lone_hits_per_channel'] = lone_hit_channel_count
        # Clear mask, don't re-use
        del mask

        # -- AFT histogram --
        aft_b = [0, 1]
        aft_hist, _ = np.histogram(peaks['area_fraction_top'], bins=n_bins, range=aft_b)
        res['aft_hist'] = aft_hist
        res['aft_bounds'] = aft_b

        # Estimate Single Electron (SE) gain
        se_hist, se_bins = np.histogram(peaks['area'], bins=n_bins,
                                        range=self.config['online_se_bounds'])
        bin_centers = (se_bins[1:] + se_bins[:1]) / 2
        res['online_se_gain'] = bin_centers[np.argmax(se_hist)]
        return res

    def area_width_hist(self, data):
        """Make area vs width 2D-hist"""
        hist, _, _ = np.histogram2d(
            np.log10(data['area']),
            np.log10(data['range_50p_area']),
            range=self.config['area_vs_width_bounds'],
            bins=self.config['area_vs_width_nbins'])
        return hist.T


@export
@strax.takes_config(
    strax.Option(
        'drift_time_vs_R2_cut_string',
        type=str,
        default='(s2_index > -1) & (s1_index > -1)',
        help='Selection (like selection_str) applied to data for '
             '"drift_time_vs_R2", cuts should be separated using "&"'),
    strax.Option(
        'drift_time_vs_R2_bounds',
        type=tuple, default=((0, 5e3), (0, 3e6)),
        help='Boundaries of histogram of drift time vs R^2.'),
    strax.Option(
        'online_event_monitor_nbins',
        type=int, default=60,
        help='Number of bins of histogram of online monitor. NB: these are 2D '
             'histograms! Will be used for: '
             'drift_time_vs_R2-histogram, '
             'drift_time_vs_s2wdith-histogram, '
             'drift_time_vs_s1aft-histogram, '
             's1area_vs_s2area-histogram, '),
    strax.Option(
        'drift_time_vs_s2width_bounds',
        type=tuple, default=((0, 3e6), (0, 1.5e4)),
        help='Boundaries of histogram of drift time vs. s2width'),
    strax.Option(
        'drift_time_vs_s1aft_bounds',
        type=tuple, default=((0, 3e6), (0, 1)),
        help='Boundaries of histogram of drift time vs. s1aft'),
    strax.Option(
        's1area_vs_s2area_bounds',
        type=tuple, default=((0, 5), (1, 7)),
        help='Boundaries of the log-log histogram of s1 area vs. s2 area'),
)
class OnlineEventMonitor(strax.Plugin):
    """
    Plugin to write data to the online-monitor. Data that is written by
    this plugin should be small such as to not overload the runs-
    database.

    This plugin takes 'event_basics'.

    Produces 'online_event_monitor' with info on the events.
    """
    depends_on = ('event_basics',)
    provides = 'online_event_monitor'
    __version__ = '0.0.1'
    data_kind = 'online_event_monitor'
    rechunk_on_save = False

    def infer_dtype(self):
        n_bins = self.config['online_event_monitor_nbins']
        bounds_drift_time_R2 = self.config['drift_time_vs_R2_bounds']

        dtype = [
            (('Start time of the chunk', 'time'),
             np.int64),
            (('End time of the chunk', 'endtime'),
             np.int64),
            (('Drift time vs R^2 histogram', 'drift_time_vs_R2_hist'),
             (np.int64, (n_bins, n_bins))),
            (('Drift time vs R^2 edges (linear space)', 'drift_time_vs_R2_bounds'),
             (np.float64, np.shape(bounds_drift_time_R2))),
            (('Drift time vs. s2width histogram', 'drift_time_vs_s2width_hist'),
             (np.int64, (n_bins, n_bins))),
            (('Drift time vs. s2width edges', 'drift_time_vs_s2width_bounds'),
             (np.float64, np.shape(self.config['drift_time_vs_s2width_bounds']))),
            (('Drift time vs. s1aft histogram', 'drift_time_vs_s1aft_hist'),
             (np.int64, (n_bins, n_bins))),
            (('Drift time vs. s1aft edges', 'drift_time_vs_s1aft_bounds'),
             (np.float64, np.shape(self.config['drift_time_vs_s1aft_bounds']))),
            (('S1 area vs. S2 area histogram', 's1area_vs_s2area_hist'),
             (np.int64, (n_bins, n_bins))),
            (('S1 area vs. S2 area edges (log-log)', 's1area_vs_s2area_bounds'),
             (np.float64, np.shape(self.config['s1area_vs_s2area_bounds']))),
        ]
        return dtype

    def compute(self, events, start, end):
        # General setup
        res = np.zeros(1, dtype=self.dtype)
        res['time'] = start
        res['endtime'] = end

        # --- Drift time vs R^2 histogram ---
        res['drift_time_vs_R2_bounds'] = self.config['drift_time_vs_R2_bounds']

        mask = _config_as_selection_str(
            self.config['drift_time_vs_R2_cut_string'], events)
        res['drift_time_vs_R2_hist'] = self.drift_time_R2_hist(events[mask])
        # Make a new mask, don't re-use
        del mask

        # drift time vs. s2 width histogram
        res['drift_time_vs_s2width_bounds'] = self.config['drift_time_vs_s2width_bounds']
        res['drift_time_vs_s2width_hist'] = self.drift_time_s2width_hist(events)

        # drift time vs. s1 aft histogram
        res['drift_time_vs_s1aft_bounds'] = self.config['drift_time_vs_s1aft_bounds']
        res['drift_time_vs_s1aft_hist'] = self.drift_time_s1aft_hist(events)

        # s1 area vs. s2 area (log-log)
        res['s1area_vs_s2area_bounds'] = self.config['s1area_vs_s2area_bounds']
        res['s1area_vs_s2area_hist'] = self.s1area_s2area_hist(events)

        # Other event properties?
        return res

    def drift_time_R2_hist(self, data):
        """Make drift time vs R^2 2D-hist"""
        hist, _, _ = np.histogram2d(
            (data['s2_x'] ** 2 + data['s2_y'] ** 2),
            data['drift_time'],
            range=self.config['drift_time_vs_R2_bounds'],
            bins=self.config['online_event_monitor_nbins'])
        return hist.T

    def drift_time_s2width_hist(self, data):
        """Make drift time vs. s2width 2D-hist"""
        hist, _, _ = np.histogram2d(
            data['drift_time'],
            data['s2_range_50p_area'],
            range=self.config['drift_time_vs_s2width_bounds'],
            bins=self.config['online_event_monitor_nbins'])
        return hist.T

    def drift_time_s1aft_hist(self, data):
        """Make drift time vs. s1aft 2D-hist"""
        hist, _, _ = np.histogram2d(
            data['drift_time'],
            data['s1_area_fraction_top'],
            range=self.config['drift_time_vs_s1aft_bounds'],
            bins=self.config['online_event_monitor_nbins'])
        return hist.T

    def s1area_s2area_hist(self, data):
        """Make s1aft vs. drift time 2D-hist"""
        hist, _, _ = np.histogram2d(
            np.log10(data['s1_area']),
            np.log10(data['s2_area']),
            range=self.config['s1area_vs_s2area_bounds'],
            bins=self.config['online_event_monitor_nbins']
        )
        return hist.T


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
