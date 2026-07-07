# This plugin has been adapted from the original MergedS2s plugin in straxen v2.2.7

from typing import Tuple

import strax
import straxen

import numpy as np
import numba

from straxen.plugins.defaults import DEFAULT_POSREC_ALGO

export, __all__ = strax.exporter()


@export
class MergedS2sVanilla(strax.OverlapWindowPlugin):
    """Merge together peaklets if peak finding favours that they would form a single peak
    instead."""

    __version__ = "1.0.3"

    depends_on: Tuple[str, ...] = (
        "peaklets",
        "peaklet_classification",
        "lone_hits",
    )
    # add "peaklet_positions_{DEFAULT_POSREC_ALGO}" if position check is desired.
    # Also uncomment the position check related lines in get_merge_instructions and compute
    data_kind = "merged_s2s"
    provides = "merged_s2s"

    s2_merge_max_duration = straxen.URLConfig(
        default=50_000,
        infer_type=False,
        help="Do not merge peaklets at all if the result would be a peak longer than this [ns]",
    )

    s2_merge_gap_thresholds_vanilla = straxen.URLConfig(
        default=((1.7, 2.65e4), (4.0, 2.6e3), (5.0, 0.0)),
        infer_type=False,
        help=(
            "Points to define maximum separation between peaklets to allow "
            "merging [ns] depending on log10 area of the merged peak\n"
            "where the gap size of the first point is the maximum gap to allow merging"
            "and the area of the last point is the maximum area to allow merging. "
            "The format is ((log10(area), max_gap), (..., ...), (..., ...))"
        ),
    )

    gain_model = straxen.URLConfig(
        infer_type=False,
        help="PMT gain model. Specify as (str(model_config), str(version), nT-->boolean",
    )

    merge_without_s1 = straxen.URLConfig(
        default=True,
        infer_type=False,
        help=(
            "If true, S1s will be igored during the merging. "
            "It's now possible for a S1 to be inside a S2 post merging"
        ),
    )

    n_top_pmts = straxen.URLConfig(type=int, help="Number of top TPC array PMTs")

    n_tpc_pmts = straxen.URLConfig(type=int, help="Number of TPC PMTs")

    sum_waveform_top_array = straxen.URLConfig(
        default=True, type=bool, help="Digitize the sum waveform of the top array separately"
    )

    merged_s2s_get_window_size_factor = straxen.URLConfig(
        default=5, type=int, track=False, help="Factor of the window size for the merged_s2s plugin"
    )

    s2_merge_dr_thresholds = straxen.URLConfig(
        default=(
            (1.51, 1.40e01),
            (1.84, 1.83e01),
            (2.18, 1.97e01),
            (2.51, 1.24e01),
            (2.84, 5.75e00),
        ),
        type=tuple,
        help=(
            "Points to define maximum weighted mean deviation of "
            "the peaklets from the main cluster [cm]\n"
            "The format is ((log10(area_top), dr), (..., ...), (..., ...))"
        ),
    )

    rm_sparse_xy = straxen.URLConfig(
        default=True, type=bool, help="Remove peaklets that are too far away in (x, y)"
    )

    use_uncertainty_weights = straxen.URLConfig(
        default=True, type=bool, help="Use uncertainty from probabilistic posrec to derive weights"
    )

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )

    merge_lone_hits = straxen.URLConfig(
        default=True,
        type=bool,
        help="Merge lone hits into merged S2s",
    )

    use_natural_break_gof = straxen.URLConfig(
        default=True, type=bool, help="Whether to use the gof field for merging"
    )

    peak_merge_gof_threshold = straxen.URLConfig(
        default=(None, ((2.5, 1.0), (5.625, 0.4))),  # The same as in peaklet plugin
        infer_type=False,
        help=(
            "Natural breaks goodness of fit/split threshold to split "
            "a peak. Specify as tuples of (log10(area), threshold)."
        ),
    )

    indicator_dtype = np.dtype(
        [(("Peaklet is merging input or peak is merged from peaklets", "merged"), bool)]
    )

    def setup(self):
        self.to_pe = self.gain_model
        self.dr_thresholds = np.array(self.s2_merge_dr_thresholds).T

    def infer_dtype(self):
        peaklet_classification_dtype = self.deps["peaklet_classification"].dtype_for(
            "peaklet_classification"
        )
        peaklets_dtype = self.deps["peaklets"].dtype_for("peaklets")
        # Include indicator_dtype - the merged field will be in the output
        # peaklet_classification now has merged field, so final dtype will have it too
        # The merged dtype is argument position dependent!
        # It must be first classification then peaklet
        # Otherwise strax will raise an error
        # when checking for the returned dtype!
        merged_s2s_dtype = strax.merged_dtype(
            (peaklet_classification_dtype, peaklets_dtype, self.indicator_dtype)
        )
        return merged_s2s_dtype

    def get_window_size(self):
        return self.merged_s2s_get_window_size_factor * (
            int(self.s2_merge_gap_thresholds_vanilla[0][1]) + self.s2_merge_max_duration
        )

    def compute(self, peaklets, lone_hits):
        if self.merge_without_s1:
            peaklets = peaklets[peaklets["type"] != 1]

        print(peaklets.dtype.names)

        if len(peaklets) <= 1:
            return np.zeros(0, dtype=self.dtype)

        gap_thresholds = self.s2_merge_gap_thresholds_vanilla
        max_gap = gap_thresholds[0][1]
        max_area = 10 ** gap_thresholds[-1][0]

        s2_gof_thresholds = self.peak_merge_gof_threshold[1]

        if max_gap < 0:
            # Do not merge at all
            return np.zeros(0, dtype=self.dtype)

        if "data_top" not in peaklets.dtype.names or "data_start" not in peaklets.dtype.names:
            # Need to add data_top field. Also add data_start (required by strax merge_peaks).
            # Note: strax's numba-compiled functions can't handle missing fields,
            # so we always include data_start even if it stays empty (zeros).
            # This is a workaround until strax properly supports optional fields.
            peak_field_w_top = strax.peak_dtype(
                n_channels=self.n_tpc_pmts,
                store_data_top=True,
                store_data_start=True,
            )
            all_field_w_top = peaklets.dtype.descr + [
                field for field in peak_field_w_top if field[0][1] not in peaklets.dtype.names
            ]
            all_field_w_top = sorted(all_field_w_top, key=lambda x: x[0][1])
            peaklets_w_field = np.zeros(len(peaklets), dtype=all_field_w_top)
            strax.copy_to_buffer(peaklets, peaklets_w_field, "_add_data_top_field")
            del peaklets
            peaklets = peaklets_w_field

        assert "data_top" in peaklets.dtype.names

        # if self.use_uncertainty_weights:
        #     name = f"position_contour_{self.default_reconstruction_algorithm}"
        #     if name not in peaklets.dtype.names:
        #         raise ValueError(f"{name} is not in the input peaklets dtype")

        # Max gap and area should be set by the gap thresholds
        # to avoid contradictions
        start_merge_at, end_merge_at = self.get_merge_instructions(
            peaklets,
            gap_thresholds=gap_thresholds,
            max_duration=self.s2_merge_max_duration,
            max_gap=max_gap,
            max_area=max_area,
            dr_thresholds=self.dr_thresholds,
            gof_thresholds=s2_gof_thresholds,
            posrec_algo=self.default_reconstruction_algorithm,
            sparse_xy=True,
            natural_break=True,
            uncertainty_weights=True,
        )

        merged_s2s = strax.merge_peaks(
            peaklets,
            start_merge_at,
            end_merge_at,
            max_buffer=int(self.s2_merge_max_duration // np.gcd.reduce(peaklets["dt"])),
        )
        merged_s2s["type"] = 2

        strax.compute_properties(merged_s2s, n_top_channels=self.n_top_pmts)

        # Updated time and length of lone_hits and sort again:
        lh = np.copy(lone_hits)
        del lone_hits
        lh_time_shift = (lh["left"] - lh["left_integration"]) * lh["dt"]
        lh["time"] = lh["time"] - lh_time_shift
        lh["length"] = lh["right_integration"] - lh["left_integration"]
        lh = strax.sort_by_time(lh)

        # Check which waveform fields are present in the output dtype
        _store_data_top = "data_top" in self.dtype_for("merged_s2s").names
        _store_data_start = "data_start" in self.dtype_for("merged_s2s").names
        n_top_pmts_if_digitize_top = self.n_top_pmts if _store_data_top else -1

        strax.add_lone_hits(
            merged_s2s,
            lh,
            self.to_pe,
            n_top_channels=n_top_pmts_if_digitize_top,
            store_data_top=_store_data_top,
            store_data_start=_store_data_start,
        )
        if len(merged_s2s) > 0:
            strax.compute_widths(merged_s2s)

        # Set merged field to True for all merged S2s (if field exists)
        if "merged" in merged_s2s.dtype.names:
            merged_s2s["merged"] = True

        if n_top_pmts_if_digitize_top <= 0:
            merged_s2s = drop_data_top_field(merged_s2s, self.dtype, "_drop_top_merged_s2s")
        return merged_s2s

    @staticmethod
    def get_merge_instructions(
        peaklets,
        gap_thresholds,
        max_duration,
        max_gap,
        max_area,
        dr_thresholds,
        gof_thresholds,
        posrec_algo,
        sparse_xy=True,
        natural_break=True,
        uncertainty_weights=True,
        sort_kind="mergesort",
    ):
        """
        Finding the group of peaklets to merge. To do this start with the
        smallest gaps and keep merging until the new, merged S2 has such a
        large area or gap to adjacent peaks that merging is not required
        anymore.
        see https://github.com/XENONnT/straxen/pull/548
        and https://github.com/XENONnT/straxen/pull/568

        :return: list of the first index of peaklet to be merged and
        list of the exclusive last index of peaklet to be merged
        """

        peaklet_starts = peaklets["time"]
        peaklet_ends = strax.endtime(peaklets)
        types = peaklets["type"]
        areas = peaklets["area"]
        # area_top = areas * peaklets["area_fraction_top"]

        # (x, y) positions of the peaklets
        # positions = np.vstack([peaklets[f"x_{posrec_algo}"], peaklets[f"y_{posrec_algo}"]]).T
        # if uncertainty_weights:
        #     contour_area = peaklets[f"position_contour_area_{posrec_algo}"]

        peaklet_gaps = peaklet_starts[1:] - peaklet_ends[:-1]
        peaklet_start_index = np.arange(len(peaklet_starts))
        peaklet_end_index = np.arange(len(peaklet_starts))

        for gap_i in np.argsort(peaklet_gaps, kind=sort_kind):
            start_idx = peaklet_start_index[gap_i]
            inclusive_end_idx = peaklet_end_index[gap_i + 1]
            sum_area = np.sum(areas[start_idx : inclusive_end_idx + 1])
            this_gap = peaklet_gaps[gap_i]

            if inclusive_end_idx < start_idx:
                raise ValueError("Something went wrong, left is bigger then right?!")

            if this_gap > max_gap:
                break
            if sum_area > max_area:
                # For very large S2s, we assume that natural breaks is taking care
                continue
            if (sum_area > 0) and (
                this_gap > merge_s2_threshold(np.log10(sum_area), gap_thresholds)
            ):
                # The merged peak would be too large
                continue

            peak_duration = peaklet_ends[inclusive_end_idx] - peaklet_starts[start_idx]
            if peak_duration >= max_duration:
                continue

            # merging = slice(start_idx, inclusive_end_idx + 1)
            # if sparse_xy:
            #     x_sel = positions[merging, 0]
            #     y_sel = positions[merging, 1]
            #     area_top_sel = area_top[merging]

            #     if uncertainty_weights:
            #         contour_sel = contour_area[merging]
            #         weights = 1.0 / contour_sel
            #     else:
            #         weights = area_top_sel

            #     dr_avg = weighted_averaged_dr(x_sel, y_sel, weights)
            #     area_top_sum = np.sum(area_top_sel)
            #     dr_threshold_ = thresholds_interpolation(np.log10(area_top_sum), dr_thresholds)

            #     if dr_avg > dr_threshold_:
            #         continue

            if natural_break:
                gof = gof_at_gap(peaklets, gap_i, peaklet_start_index, peaklet_end_index)

                # high gof means that the split is good, so we do not merge
                if gof > merge_s2_threshold(np.log10(sum_area), gof_thresholds):
                    continue

            # Merge gap in other words this means p @ gap_i and p @gap_i + 1 share the same
            # start, end and area:
            peaklet_start_index[start_idx : inclusive_end_idx + 1] = peaklet_start_index[start_idx]
            peaklet_end_index[start_idx : inclusive_end_idx + 1] = peaklet_end_index[
                inclusive_end_idx
            ]

        start_merge_at = np.unique(peaklet_start_index)
        end_merge_at = np.unique(peaklet_end_index)
        if not len(start_merge_at) == len(end_merge_at):
            raise ValueError("inconsistent start and end merge instructions")

        merge_start, merge_stop_exclusive = _filter_s1_starts(start_merge_at, types, end_merge_at)

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
def merge_s2_threshold(log_area, thresholds):
    """Return gap threshold for log_area of the merged S2 with linear interpolation given the points
    in gap_thresholds and gof_thresholds.

    :param log_area: Log 10 area of the merged S2
    :param gap_thresholds: tuple (n, 2) of fix points for interpolation.

    """
    for i, (a1, g1) in enumerate(thresholds):
        if log_area < a1:
            if i == 0:
                return g1
            a0, g0 = thresholds[i - 1]
            return (log_area - a0) * (g1 - g0) / (a1 - a0) + g0
    return thresholds[-1][1]


def drop_data_top_field(peaklets, goal_dtype, _name_function="_drop_data_top_field"):
    """Return peaklets without the data_top field."""
    peaklets_without_top_field = np.zeros(len(peaklets), dtype=goal_dtype)
    strax.copy_to_buffer(peaklets, peaklets_without_top_field, _name_function)
    del peaklets
    return peaklets_without_top_field


@numba.njit(cache=True)
def thresholds_interpolation(log_area, thresholds):
    """Return threshold for log_area of the merged S2 with linear interpolation given the points in
    thresholds.

    :param log_area: Log 10 area of the merged S2
    :param thresholds: tuple (n, 2) of fix points for interpolation.

    """
    if log_area < thresholds[0, 0]:
        return thresholds[1, 0]
    if log_area > thresholds[0, -1]:
        return thresholds[1, -1]
    return np.interp(log_area, thresholds[0], thresholds[1])


@numba.njit(cache=True, nogil=True)
def weighted_averaged_dr(x, y, weights):
    """Weighted average deviation from weighted average (x, y)"""
    mask = weights > 0
    mask &= ~np.isnan(x)
    mask &= ~np.isnan(y)
    # do not merge any S2 looks weird
    if not np.all(mask):
        return np.nan
    x_avg = np.average(x[mask], weights=weights[mask])
    y_avg = np.average(y[mask], weights=weights[mask])
    dr = np.sqrt((x - x_avg) ** 2 + (y - y_avg) ** 2)
    dr_avg = np.average(dr[mask], weights=weights[mask])
    return dr_avg


@numba.njit(cache=True, nogil=True)
def total_variance(peaklets, start_idx, end_idx, time_gap, from_right_to_left=False):
    """Accumulate weighted time moments over a contiguous peaklet range."""
    total_w = 0.0
    total_w_t = 0.0
    total_w_t2 = 0.0

    cum_w_var_out = []
    _w = []

    if from_right_to_left:
        peak_range = np.arange(start_idx, end_idx + 1)[::-1]
    else:
        peak_range = np.arange(start_idx, end_idx + 1)
    for pi in peak_range:

        waveform = peaklets[pi]["data"]
        length = peaklets[pi]["length"]
        dt = peaklets[pi]["dt"]
        t0 = peaklets[pi]["time"]

        if from_right_to_left:
            loop_range = np.arange(length)[::-1]
        else:
            loop_range = np.arange(length)
        for i in loop_range:
            charge = waveform[i]

            if charge <= 0.0:
                charge = 0.0

            t = (t0 - time_gap) + dt * (i + 0.5)
            w = charge
            _w.append(w / dt)
            total_w += w
            total_w_t += w * t
            total_w_t2 += w * t * t

            if total_w <= 0.0:
                total_w_var = 0.0
            else:
                total_w_var = total_w_t2 - (total_w_t * total_w_t) / total_w

            cum_w_var_out.append(total_w_var)
        total_w_var = cum_w_var_out[-1] if len(cum_w_var_out) > 0 else 0.0

    return np.array(cum_w_var_out), total_w_var, np.array(_w)


@numba.njit(cache=True, nogil=True)
def gof_at_gap(peaklets, gap_i, peaklet_start_idx, peaklet_end_idx):
    """Compute the left-side weighted variance contribution for one split.

    This avoids the previous two-pass mean/variance calculation. The same result can be computed
    directly from the combined weighted moments.

    """
    time_gap = peaklets[gap_i + 1]["time"]

    left_w_sum_variance, merged_w_sum_variance, left_norm_w = total_variance(
        peaklets,
        peaklet_start_idx[gap_i],
        peaklet_end_idx[gap_i + 1],
        time_gap,
    )

    right_w_sum_variance, ___, __ = total_variance(
        peaklets,
        peaklet_start_idx[gap_i],
        peaklet_end_idx[gap_i + 1],
        time_gap,
        from_right_to_left=True,
    )

    gof_array = np.empty(len(left_w_sum_variance), dtype=np.float64)
    right_rev = right_w_sum_variance[::-1]
    for i in range(len(left_w_sum_variance)):
        gof_array[i] = 1.0 - (left_w_sum_variance[i] + right_rev[i]) / merged_w_sum_variance

    lw_max = np.max(left_norm_w)
    if lw_max > 0:
        gof_array = gof_array * (1.0 - left_norm_w / lw_max)  # low_split
    gof = np.max(gof_array)

    return gof
