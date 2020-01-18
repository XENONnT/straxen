import numba
import numpy as np

import strax
import straxen

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('peaklet_gap_threshold', default=350,
                 help="No hits for this many ns triggers a new peak"),
    strax.Option('peak_left_extension', default=30,
                 help="Include this many ns left of hits in peaks"),
    strax.Option('peak_right_extension', default=30,
                 help="Include this many ns right of hits in peaks"),
    strax.Option('peak_min_pmts', default=2,
                 help="Minimum contributing PMTs needed to define a peak"),
    strax.Option('peaklet_split_min_height', default=25,
                 help="Minimum height in PE above a local sum waveform"
                      "minimum, on either side, to trigger a split"),
    strax.Option('peaklet_split_min_ratio', default=4,
                 help="Minimum ratio between local sum waveform"
                      "minimum and maxima on either side, to trigger a split"),
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"),
    strax.Option('to_pe_file',
                 default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',
                 help='Link to the to_pe conversion factors'),
    strax.Option('tight_coincidence_window_left', default=50,
                 help="Time range left of peak center to call "
                      "a hit a tight coincidence (ns)"),
    strax.Option('tight_coincidence_window_right', default=50,
                 help="Time range right of peak center to call "
                      "a hit a tight coincidence (ns)"))
class Peaklets(strax.Plugin):
    depends_on = ('records',)
    data_kind = 'peaklets'
    provides = 'peaklets'
    parallel = 'process'
    rechunk_on_save = True

    __version__ = '0.1.1'

    def infer_dtype(self):
        self.to_pe = straxen.get_to_pe(self.run_id, self.config['to_pe_file'])
        return strax.peak_dtype(n_channels=len(self.to_pe))

    def compute(self, records):
        r = records

        hits = strax.find_hits(r)

        # Remove hits in zero-gain channels
        # they should not affect the clustering!
        hits = hits[self.to_pe[hits['channel']] != 0]

        hits = strax.sort_by_time(hits)

        # Use peaklet gap threshold for initial clustering
        # based on gaps between hits
        peaklets = strax.find_peaks(
            hits, self.to_pe,
            gap_threshold=self.config['peaklet_gap_threshold'],
            left_extension=self.config['peak_left_extension'],
            right_extension=self.config['peak_right_extension'],
            min_channels=self.config['peak_min_pmts'],
            result_dtype=self.dtype)

        strax.sum_waveform(peaklets, r, self.to_pe)

        # split based on local minima
        peaklets = strax.split_peaks(
            peaklets, r, self.to_pe,
            min_height=self.config['peaklet_split_min_height'],
            min_ratio=self.config['peaklet_split_min_ratio'])

        # Need widths for pseudo-classification
        strax.compute_widths(peaklets)

        # Compute tight coincidence level.
        # Making this a separate plugin would
        # (a) doing hitfinding yet again (or storing hits)
        # (b) increase strax memory usage / max_messages,
        #     possibly due to its currently primitive scheduling.
        hit_max_times = np.sort(
            hits['time']
            + hits['dt'] * hit_max_sample(records, hits))
        peaklet_max_times = (
                peaklets['time']
                + np.argmax(peaklets['data'], axis=1) * peaklets['dt'])
        peaklets['tight_coincidence'] = get_tight_coin(
            hit_max_times,
            peaklet_max_times,
            self.config['tight_coincidence_window_left'],
            self.config['tight_coincidence_window_right'])

        if self.config['diagnose_sorting'] and len(r):
            assert np.diff(r['time']).min(initial=1) >= 0, "Records not sorted"
            assert np.diff(hits['time']).min(initial=1) >= 0, "Hits not sorted"
            assert np.all(peaklets['time'][1:]
                          >= strax.endtime(peaklets)[:-1]), "Peaks not disjoint"

        return peaklets


@export
@strax.takes_config(
    strax.Option('s1_max_rise_time', default=60,
                 help="Maximum S1 rise time for < 100 PE [ns]"),
    strax.Option('s1_max_rise_time_post100', default=150,
                 help="Maximum S1 rise time for > 100 PE [ns]"),
    strax.Option('s1_min_coincidence', default=3,
                 help="Minimum tight coincidence necessary to make an S1"),
    strax.Option('s2_min_pmts', default=4,
                 help="Minimum number of PMTs contributing to an S2"))
class PeakletClassification(strax.Plugin):
    """Classify peaklets as unknown, S1, or S2."""
    provides = 'peaklet_classification'
    depends_on = ('peaklets',)
    parallel = True
    dtype = [('type', np.int8, 'Classification of the peak(let)')]
    __version__ = '0.1.0'

    def compute(self, peaklets):
        peaks = peaklets

        ptype = np.zeros(len(peaklets), dtype=np.int8)

        # Properties needed for classification. Bit annoying these computations
        # are duplicated in peak_basics curently...
        rise_time = -peaks['area_decile_from_midpoint'][:, 1]
        n_channels = (peaks['area_per_channel'] > 0).sum(axis=1)

        is_s1 = (
           (rise_time <= self.config['s1_max_rise_time'])
            | ((rise_time <= self.config['s1_max_rise_time_post100'])
               & (peaks['area'] > 100)))
        is_s1 &= peaks['tight_coincidence'] >= self.config['s1_min_coincidence']
        ptype[is_s1] = 1

        is_s2 = n_channels >= self.config['s2_min_pmts']
        is_s2[is_s1] = False
        ptype[is_s2] = 2

        return dict(type=ptype)


FAKE_MERGED_S2_TYPE = -42

@export
@strax.takes_config(
    strax.Option('s2_merge_max_area', default=5000.,
                 help="Merge peaklet cluster only if area < this [PE]"),
    strax.Option('s2_merge_max_gap', default=3500,
                 help="Maximum separation between peaklets to allow merging [ns]"),
    strax.Option('s2_merge_max_duration', default=15_000,
                 help="Do not merge peaklets at all if the result would be a peak "
                      "longer than this [ns]"))
class MergedS2s(strax.OverlapWindowPlugin):
    """Merge together peaklets if we believe they form a single peak instead
    """
    depends_on = ('peaklets', 'peaklet_classification')
    data_kind = 'merged_s2s'
    provides = 'merged_s2s'

    # Keep chunk mapping the same as peaks
    # If we make one huge chunk, peak building
    # will be very RAM_intensive
    rechunk_on_save = False

    def infer_dtype(self):
        return self.deps['peaklets'].dtype_for('peaklets')

    def get_window_size(self):
        return 5 * (self.config['s2_merge_max_gap']
                    + self.config['s2_merge_max_duration'])

    def compute(self, peaklets):
        if not len(peaklets):
            return peaklets[:0]

        if self.config['s2_merge_max_gap'] < 0:
            # Do not merge at all
            merged_s2s = np.zeros(0, dtype=peaklets.dtype)
        else:
            # Find all groups of peaklets separated by < the gap
            cluster_starts, cluster_stops = strax.find_peak_groups(
                peaklets,
                self.config['s2_merge_max_gap'])

            start_merge_at, end_merge_at = self.get_merge_instructions(
                peaklets['time'], strax.endtime(peaklets),
                areas=peaklets['area'],
                types=peaklets['type'],
                cluster_starts=cluster_starts,
                cluster_stops=cluster_stops,
                max_duration=self.config['s2_merge_max_duration'],
                max_area=self.config['s2_merge_max_area'])

            merged_s2s = strax.merge_peaks(
                peaklets,
                start_merge_at, end_merge_at,
                max_buffer=int(self.config['s2_merge_max_duration']
                               // peaklets['dt'].min()))
            merged_s2s['type'] = 2
            strax.compute_widths(merged_s2s)

        if len(merged_s2s) == 0:
            # Strax does not handle the case of no merged S2s well
            # If there are none in the entire dataset, it will just keep
            # waiting in Peaks forever.
            # Thus, this ugly hack of passing a single fake merged S2
            # in the middle of the chunk, which is removed later
            merged_s2s = np.zeros(1, merged_s2s.dtype)
            q = merged_s2s[0]
            q['type'] = FAKE_MERGED_S2_TYPE
            q['time'] = (peaklets[0]['time']
                         + strax.endtime(peaklets[0])) / 2
            q['dt'] = 1
        return merged_s2s

    @staticmethod
    @numba.njit(cache=True, nogil=True)
    def get_merge_instructions(
            peaklet_starts, peaklet_ends, areas, types,
            cluster_starts, cluster_stops,
            max_duration, max_area):
        start_merge_at = np.zeros(len(cluster_starts), dtype=np.int32)
        end_merge_at = np.zeros(len(cluster_starts), dtype=np.int32)
        n_to_merge = 0
        left_i = 0

        for cluster_i, cluster_start in enumerate(cluster_starts):
            cluster_stop = cluster_stops[cluster_i]

            if cluster_stop - cluster_start > max_duration:
                continue

            # Recover left and right indices of the clusters.
            # stops are inclusive for a few lines... sorry...
            while peaklet_starts[left_i] < cluster_start:
                left_i += 1
            right_i = left_i
            while peaklet_ends[right_i] < cluster_stop:
                right_i += 1

            if left_i == right_i:
                # One peak, nothing to merge
                continue

            if types[left_i] != 2:
                # Doesn't start with S2: do not merge
                continue

            right_i += 1   # From here on, right_i is exclusive

            if areas[left_i:right_i].sum() > max_area:
                continue

            start_merge_at[n_to_merge] = left_i
            end_merge_at[n_to_merge] = right_i
            n_to_merge += 1

        return start_merge_at[:n_to_merge], end_merge_at[:n_to_merge]


@export
@strax.takes_config(
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"))
class Peaks(strax.Plugin):
    # NB: merged_s2s must come first, otherwise a chunk could end
    # in the middle of a merged peak -> bad stuff
    depends_on = ('merged_s2s', 'peaklets', 'peaklet_classification')
    data_kind = 'peaks'
    provides = 'peaks'
    parallel = True
    save_when = strax.SaveWhen.NEVER

    __version__ = '0.1.1'

    def infer_dtype(self):
        return self.deps['peaklets'].dtype_for('peaklets')

    def compute(self, peaklets, merged_s2s):
        # Remove fake merged S2s from dirty hack, see above
        merged_s2s = merged_s2s[merged_s2s['type'] != FAKE_MERGED_S2_TYPE]

        peaks = strax.replace_merged(peaklets, merged_s2s)

        if self.config['diagnose_sorting']:
            assert np.all(np.diff(peaks['time']) >= 0), "Peaks not sorted"
            assert np.all(peaks['time'][1:]
                          >= strax.endtime(peaks)[:-1]), "Peaks not disjoint"
        return peaks



@numba.jit(nopython=True, nogil=True, cache=True)
def get_tight_coin(hit_max_times, peak_max_times, left, right):
    """Calculates the tight coincidence

    Defined by number of hits within a specified time range of the
    the peak's maximum amplitude.
    Imitates tight_coincidence variable in pax:
    github.com/XENON1T/pax/blob/master/pax/plugins/peak_processing/BasicProperties.py
    """
    left_hit_i = 0
    n_coin = np.zeros(len(peak_max_times), dtype=np.int16)

    # loop over peaks
    for p_i, p_t in enumerate(peak_max_times):

        # loop over hits starting from the last one we left at
        for left_hit_i in range(left_hit_i, len(hit_max_times)):

            # if the hit is in the window, its a tight coin
            d = hit_max_times[left_hit_i] - p_t
            if (-left < d) & (d < right):
                n_coin[p_i] += 1

            # stop the loop when we know we're outside the range
            if d > right:
                break

    return n_coin


@numba.njit(cache=True, nogil=True)
def hit_max_sample(records, hits):
    """Return the index of the maximum sample for hits"""
    result = np.zeros(len(hits), dtype=np.int16)
    for i, h in enumerate(hits):
        r = records[h['record_i']]
        w = r['data'][h['left']:h['right']]
        result[i] = np.argmax(w)
    return result
