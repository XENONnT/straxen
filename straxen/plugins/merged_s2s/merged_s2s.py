import strax
import straxen
from straxen.plugins.peaklets.peaklets import drop_data_top_field

import numpy as np
import numba


export, __all__ = strax.exporter()


@export
class MergedS2s(strax.OverlapWindowPlugin):
    """
    Merge together peaklets if peak finding favours that they would
    form a single peak instead.
    """
    __version__ = '1.0.0'

    depends_on = ('peaklets', 'peaklet_classification', 'lone_hits')
    data_kind = 'merged_s2s'
    provides = 'merged_s2s'

    s2_merge_max_duration = straxen.URLConfig(
        default=50_000, infer_type=False,
        help="Do not merge peaklets at all if the result would be a peak "
             "longer than this [ns]")

    s2_merge_gap_thresholds = straxen.URLConfig(
        default=((1.7, 2.65e4), (4.0, 2.6e3), (5.0, 0.)),
        infer_type=False,
        help="Points to define maximum separation between peaklets to allow "
             "merging [ns] depending on log10 area of the merged peak\n"
             "where the gap size of the first point is the maximum gap to allow merging"
             "and the area of the last point is the maximum area to allow merging. "
             "The format is ((log10(area), max_gap), (..., ...), (..., ...))"
    )

    gain_model = straxen.URLConfig(
        infer_type=False,
        help='PMT gain model. Specify as '
             '(str(model_config), str(version), nT-->boolean')

    merge_without_s1 = straxen.URLConfig(
        default=True, infer_type=False,
        help="If true, S1s will be igored during the merging. "
             "It's now possible for a S1 to be inside a S2 post merging")

    n_top_pmts = straxen.URLConfig(
        type=int,
        help="Number of top TPC array PMTs")

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help='Number of TPC PMTs')

    sum_waveform_top_array = straxen.URLConfig(
        default=True,
        type=bool,
        help='Digitize the sum waveform of the top array separately'
    )

    def setup(self):
        self.to_pe = self.gain_model

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps['peaklets'].dtype_for('peaklets'))

    def get_window_size(self):
        return 5 * (int(self.s2_merge_gap_thresholds[0][1])
                    + self.s2_merge_max_duration)

    def compute(self, peaklets, lone_hits):
        if self.merge_without_s1:
            peaklets = peaklets[peaklets['type'] != 1]

        if len(peaklets) <= 1:
            return np.zeros(0, dtype=self.dtype)

        gap_thresholds = self.s2_merge_gap_thresholds
        max_gap = gap_thresholds[0][1]
        max_area = 10 ** gap_thresholds[-1][0]

        if max_gap < 0:
            # Do not merge at all
            return np.zeros(0, dtype=self.dtype)

        if 'data_top' not in peaklets.dtype.names:
            peaklets_w_field = np.zeros(len(peaklets), dtype=strax.peak_dtype(n_channels=self.n_tpc_pmts, digitize_top=True))
            strax.copy_to_buffer(peaklets, peaklets_w_field, '_add_data_top_field')
            del peaklets
            peaklets = peaklets_w_field

        # Max gap and area should be set by the gap thresholds
        # to avoid contradictions
        start_merge_at, end_merge_at = self.get_merge_instructions(
            peaklets['time'], strax.endtime(peaklets),
            areas=peaklets['area'],
            types=peaklets['type'],
            gap_thresholds=gap_thresholds,
            max_duration=self.s2_merge_max_duration,
            max_gap=max_gap,
            max_area=max_area,
        )
        assert 'data_top'  in peaklets.dtype.names

        merged_s2s = strax.merge_peaks(
            peaklets,
            start_merge_at, end_merge_at,
            max_buffer=int(self.s2_merge_max_duration // np.gcd.reduce(peaklets['dt'])),
        )
        merged_s2s['type'] = 2

        # Updated time and length of lone_hits and sort again:
        lh = np.copy(lone_hits)
        del lone_hits
        lh_time_shift = (lh['left'] - lh['left_integration']) * lh['dt']
        lh['time'] = lh['time'] - lh_time_shift
        lh['length'] = (lh['right_integration'] - lh['left_integration'])
        lh = strax.sort_by_time(lh)

        # If sum_waveform_top_array is false, don't digitize the top array
        n_top_pmts_if_digitize_top = self.n_top_pmts if self.sum_waveform_top_array else -1
        strax.add_lone_hits(merged_s2s, lh, self.to_pe, n_top_channels=n_top_pmts_if_digitize_top)

        strax.compute_widths(merged_s2s)

        if n_top_pmts_if_digitize_top <= 0:
            merged_s2s = drop_data_top_field(merged_s2s, self.dtype, '_drop_top_merged_s2s')
        return merged_s2s

    @staticmethod
    @numba.njit(cache=True, nogil=True)
    def get_merge_instructions(
            peaklet_starts, peaklet_ends, areas, types,
            gap_thresholds, max_duration, max_gap, max_area):
        """
        Finding the group of peaklets to merge. To do this start with the
        smallest gaps and keep merging until the new, merged S2 has such a
        large area or gap to adjacent peaks that merging is not required
        anymore.
        see https://github.com/XENONnT/straxen/pull/548 and https://github.com/XENONnT/straxen/pull/568

        :returns: list of the first index of peaklet to be merged and
        list of the exclusive last index of peaklet to be merged
        """

        peaklet_gaps = peaklet_starts[1:] - peaklet_ends[:-1]
        peaklet_start_index = np.arange(len(peaklet_starts))
        peaklet_end_index = np.arange(len(peaklet_starts))

        for gap_i in np.argsort(peaklet_gaps):
            start_idx = peaklet_start_index[gap_i]
            inclusive_end_idx = peaklet_end_index[gap_i + 1]
            sum_area = np.sum(areas[start_idx:inclusive_end_idx + 1])
            this_gap = peaklet_gaps[gap_i]

            if inclusive_end_idx < start_idx:
                raise ValueError('Something went wrong, left is bigger then right?!')

            if this_gap > max_gap:
                break
            if sum_area > max_area:
                # For very large S2s, we assume that natural breaks is taking care
                continue
            if (sum_area > 0) and (
                    this_gap > merge_s2_threshold(np.log10(sum_area),
                                                  gap_thresholds)):
                # The merged peak would be too large
                continue

            peak_duration = (peaklet_ends[inclusive_end_idx] - peaklet_starts[start_idx])
            if peak_duration >= max_duration:
                continue

            # Merge gap in other words this means p @ gap_i and p @gap_i + 1 share the same
            # start, end and area:
            peaklet_start_index[start_idx:inclusive_end_idx + 1] = peaklet_start_index[start_idx]
            peaklet_end_index[start_idx:inclusive_end_idx + 1] = peaklet_end_index[inclusive_end_idx]

        start_merge_at = np.unique(peaklet_start_index)
        end_merge_at = np.unique(peaklet_end_index)
        if not len(start_merge_at) == len(end_merge_at):
            raise ValueError('inconsistent start and end merge instructions')

        merge_start, merge_stop_exclusive = _filter_s1_starts(
            start_merge_at, types, end_merge_at)

        return merge_start, merge_stop_exclusive


@numba.njit(cache=True, nogil=True)
def _filter_s1_starts(start_merge_at, types, end_merge_at):
    for start_merge_idx, _ in enumerate(start_merge_at):
        while types[start_merge_at[start_merge_idx]] != 2:
            if end_merge_at[start_merge_idx] - start_merge_at[start_merge_idx] <= 1:
                break
            start_merge_at[start_merge_idx] += 1

    start_merge_with_s2 = types[start_merge_at] == 2
    merges_at_least_two_peaks = end_merge_at - start_merge_at >= 1

    keep_merges = start_merge_with_s2 & merges_at_least_two_peaks
    return start_merge_at[keep_merges], end_merge_at[keep_merges] + 1


@numba.njit(cache=True, nogil=True)
def merge_s2_threshold(log_area, gap_thresholds):
    """Return gap threshold for log_area of the merged S2
    with linear interpolation given the points in gap_thresholds
    :param log_area: Log 10 area of the merged S2
    :param gap_thresholds: tuple (n, 2) of fix points for interpolation
    """
    for i, (a1, g1) in enumerate(gap_thresholds):
        if log_area < a1:
            if i == 0:
                return g1
            a0, g0 = gap_thresholds[i - 1]
            return (log_area - a0) * (g1 - g0) / (a1 - a0) + g0
    return gap_thresholds[-1][1]
