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
    strax.Option('s1like_max_rise_time', default=70,
                 help="Maximum S1 rise time [ns]"),
    strax.Option('s1like_max_90p_width', default=300,
                 help="Minimum tight coincidence necessary to make an S1"))
class PeakletClassification(strax.Plugin):
    depends_on = ('peaklets',)
    data_kind = 'peaklets'
    provides = 'peaklet_classification'
    parallel = 'process'
    rechunk_on_save = True
    dtype = [('type', np.int8, 'Classification of the peaklet.')]

    __version__ = '0.1.0'

    def compute(self, peaklets):
        # Tag S1-like peaks based on rise time
        # and 90p width. Return array of len(p)
        # where 1s are S1-like and everything else is 0.
        result = np.zeros(
            len(peaklets),
            dtype=self.dtype)

        # rise-time requirement
        is_s1like = -peaklets['area_decile_from_midpoint'][:, 1] <= self.config['s1like_max_rise_time']
        # 90p width requirement
        is_s1like &= peaklets['width'][:, 9] <= self.config['s1like_max_90p_width']
        result[is_s1like] = 1
        return result


@export
@strax.takes_config(
    strax.Option('peak_gap_threshold', default=3500,
                 help="No hits for this many ns triggers a new peak"),
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"),
    strax.Option('to_pe_file',
                 default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',
                 help='Link to the to_pe conversion factors'))
class PeaksFromPeaklets(strax.Plugin):
    depends_on = ('peaklets', 'peaklet_classification')
    data_kind = 'peaks'
    provides = 'peaks'
    parallel = 'process'
    rechunk_on_save = True

    __version__ = '0.1.0'

    def infer_dtype(self):
        self.to_pe = straxen.get_to_pe(self.run_id, self.config['to_pe_file'])
        return strax.peak_dtype(n_channels=len(self.to_pe))

    def compute(self, peaklets):
        # Separate peaklets that are going to be merged:
        # only those that are not S1-like
        t0, t1 = strax.find_peak_groups(
            peaklets[peaklets['type'] == 0],
            self.config['peak_gap_threshold'])

        # max samples is max-deltat / min-sample-dt
        merged_peak_max_samples = int((t1 - t0).max() / peaklets['dt'].min())

        merge_with_next = get_merge_with_next(peaklets, t0, t1)

        peaks = merge_peaks(
            peaklets,
            merge_with_next,
            max_buffer=(2 * merged_peak_max_samples))

        strax.compute_widths(peaks)

        if self.config['diagnose_sorting']:
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


@export
def get_start_end(merge_with_next):
    """Return (start, end) index arrays for peaks to be merged
    :param merge_with_next: array of 0's and 1's. 1 indicates do merge with
    next peak, 0 the opposite.
    :return: start_merge_at (array), end_merge_at (array);
        start_merge_at indicating index to start
        end_merge_at indicating index to end (note that merge thus extends to
        the peak before these indices)
    """
    if not len(merge_with_next) or len(merge_with_next) <= 1:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    assert merge_with_next[-1] != 1, \
        "Trying to merge last peak to a non-existing next peak"

    end_merge = merge_with_next[:-1] & ~merge_with_next[1:]
    start_merge = merge_with_next[1:] & ~merge_with_next[:-1]

    if merge_with_next[0] == 1:
        start_merge[0] = merge_with_next[0]

    end_merge_at = np.where(end_merge)[0] + 2
    start_merge_at = np.where(start_merge)[0] + 1
    if merge_with_next[0] == 1 and start_merge_at[0] == 1:
        start_merge_at[0] = 0

    return start_merge_at, end_merge_at


def get_merge_with_next(peaks, t0, t1):
    """Decide which peaks to merge with the following peak

    :param peaks: Record array of peak dtype. Not modified.
    :param t0: Numpy array of start times of new peak intervals.
    :param t1: Numpy array of end times of new peak intervals.

    For each interval containing n > 1 peaklets, the first n - 1
    are assigned to be merged with their following peaks. A numpy
    array with len(peaks) is returned with 1 at the indices of peaks
    to be merged-with-next, and 0 elsewhere.
    """
    end_times = strax.endtime(peaks)
    merge_with_next = np.zeros(len(peaks), dtype=np.int8)
    for i in range(len(t0)):
        merge_with_next[np.where((peaks['time'] >= t0[i])
                                 & (end_times <= t1[i]))[0][:-1]] = 1
    return merge_with_next


def merge_peaks(peaks, merge_with_next, max_buffer=int(1e5)):
    """Merge specified peaks with their neighbors

    :param peaks: Record array of strax peak dtype.
    :param merge_with_next: Array where 1s indicate which peaks to merge
    with their following neighbors.
    :param max_buffer: Maximum number of samples in the sum_waveforms of
    the resulting peaks (after merging).

    Peaks must be constructed based on the properties of constituent peaks,
    it being too time-consuming to revert to records/hits.
    """
    # Find merge start / end peaks
    start_merge_at, end_merge_at = get_start_end(merge_with_next)

    assert len(start_merge_at) == len(end_merge_at)
    new_peaks = np.zeros(len(start_merge_at),
                         strax.peak_dtype(n_channels=len(peaks[0]['saturated_channel'])))

    # Do the merging. Make sure to numbafy this when done
    buffer = np.zeros(max_buffer, dtype=np.float32)

    for new_i, new_p in enumerate(new_peaks):

        common_dt = 10  # TODO: pick least common denominator

        old_peaks = peaks[start_merge_at[new_i]:end_merge_at[new_i]]
        first_peak, last_peak = old_peaks[0], old_peaks[-1]
        new_p['time'] = first_peak['time']
        new_p['channel'] = first_peak['channel']

        # re-zero relevant part of buffer (overkill? not sure if
        # this saves much time)
        buffer[:min(
            int(
                (
                        last_peak['time']
                        + (last_peak['length'] * old_peaks['dt'].max())
                        - first_peak['time']) / common_dt
            ),
            len(buffer)
        )] = 0

        for p in old_peaks:
            # Upsample the sum waveform into the buffer
            upsample = p['dt'] // common_dt
            n_after = p['length'] * upsample
            i0 = (p['time'] - new_p['time']) // common_dt
            buffer[i0: i0 + n_after] = \
                np.repeat(p['data'][:p['length']], upsample) / upsample

            # Handle the other peak attributes
            new_p['area'] += p['area']
            new_p['area_per_channel'] += p['area_per_channel']
            new_p['n_hits'] += p['n_hits']
            new_p['saturated_channel'][p['saturated_channel'] == 1] = 1

        new_p['dt'] = common_dt
        new_p['length'] = (
                                  last_peak['time']
                                  + (last_peak['dt'] * last_peak['length'])
                                  - new_p['time']
                          ) / common_dt
        new_p['n_saturated_channels'] = new_p['saturated_channel'].sum()
        new_p['tight_coincidence'] = old_peaks['tight_coincidence'][
            old_peaks['data'].max(axis=1).argmax()
        ]

        # Downsample the buffer into new_p['data']
        strax.store_downsampled_waveform(new_p, buffer)

    # find which peaklets now belong to a merged peak
    merged_inds = merge_with_next | np.pad(
        merge_with_next[:-1],
        (1, 0),
        'constant',
        constant_values=(0, 0)
    )
    unmerged_peaks = peaks[merged_inds == 0]
    # No easy way to remove a column by name? Had to rebuild in order to
    # hstack due to potentially different dtypes (additional column 'type')
    remade_unmerged_peaks = np.zeros(len(unmerged_peaks),
                                     strax.peak_dtype(n_channels=len(peaks[0]['saturated_channel'])))
    for field in unmerged_peaks.dtype.names:
        if field in remade_unmerged_peaks.dtype.names:
            remade_unmerged_peaks[field] = unmerged_peaks[field]
    all_peaks = np.hstack((new_peaks, remade_unmerged_peaks))
    return strax.sort_by_time(all_peaks)
