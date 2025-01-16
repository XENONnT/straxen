from typing import Tuple, Dict, Union
import numpy as np
from tqdm import tqdm
from scipy.stats import norm, poisson
import numba
import strax
import straxen
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO, WIDE_XYPOS_S2_TYPE
from straxen.plugins.peaklets.peaklets import drop_data_field


export, __all__ = strax.exporter()


@export
class MergedS2s(strax.OverlapWindowPlugin):
    """Merge together peaklets if peak finding favours that they would form a single peak instead.

    Reference: xenon:xenonnt:analysis:s2_merging_time_position

    """

    __version__ = "2.0.0"

    depends_on: Tuple[str, ...] = (
        "peaklets",
        f"peaklet_positions_{DEFAULT_POSREC_ALGO}",
        "peaklet_classification",
        "lone_hits",
    )
    provides: Union[Tuple[str, ...], str] = ("merged_s2s", "enhanced_peaklet_classification")
    data_kind: Union[Dict[str, str], str] = dict(
        merged_s2s="merged_s2s", enhanced_peaklet_classification="peaklets"
    )

    n_tpc_pmts = straxen.URLConfig(type=int, help="Number of TPC PMTs")

    n_top_pmts = straxen.URLConfig(type=int, help="Number of top TPC array PMTs")

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )

    s2_merge_max_duration = straxen.URLConfig(
        default=50_000,
        type=int,
        infer_type=False,
        help="Do not merge peaklets at all if the result would be a peak longer than this [ns]",
    )

    s2_merge_gap_thresholds = straxen.URLConfig(
        default=(
            (1.84, 2.84e04),
            (2.18, 2.37e04),
            (2.51, 1.97e04),
            (2.84, 1.83e04),
            (3.18, 1.72e04),
            (3.51, 1.89e04),
            (3.84, 1.95e04),
            (4.18, 1.63e04),
            (4.51, 1.21e04),
            (4.84, 0.00e00),
        ),
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

    rough_seg = straxen.URLConfig(
        default=30, type=(int, float), help="Rough single electron gain [PE/e]"
    )

    sigma_seg = straxen.URLConfig(
        default=6.5,
        type=(int, float),
        help="Standard deviation of the single electron gain [PE/e]",
    )

    rough_min_sigma = straxen.URLConfig(
        default=1e2, type=(int, float), help="Minimum sigma for the merged peaks"
    )

    rough_max_sigma = straxen.URLConfig(
        default=2e4, type=(int, float), help="Maximum sigma for the merged peaks"
    )

    rough_sigma_bins = straxen.URLConfig(
        default=10, type=int, help="Number of bins for sigma of merged peaks"
    )

    rough_mu_bins = straxen.URLConfig(
        default=10, type=int, help="Number of bins for mu of merged peaks"
    )

    poisson_max_mu = straxen.URLConfig(
        default=25, type=int, help="When to switch from Poisson to normal distribution"
    )

    poisson_survival_ratio = straxen.URLConfig(
        default=1e-4,
        type=float,
        help=(
            "Survival ratio for Poisson distribution. The PMF smaller than this will be ignored."
        ),
    )

    normal_max_sigma = straxen.URLConfig(
        default=7,
        type=(int, float),
        help="Maximum sigma for the normal distribution CDF panel",
    )

    normal_n_bins = straxen.URLConfig(
        default=501,
        type=int,
        help="Number of bins for the normal distribution CDF panel",
    )

    maxexp = straxen.URLConfig(
        default=10,
        type=(int, float),
        help="Maximum exponent for the posterior to keep numerical stability",
    )

    p_threshold = straxen.URLConfig(
        default=1e-2,
        type=(int, float),
        help="Threshold for the p-value of time-density merging",
    )

    dr_threshold = straxen.URLConfig(
        default=14.0,
        type=(int, float),
        help="Threshold for the weighted mean deviation of the peaklets from the main cluster [cm]",
    )

    use_bayesian_merging = straxen.URLConfig(default=True, type=bool, help="Use Bayesian merging")

    use_uncertainty_weights = straxen.URLConfig(
        default=True, 
        type=bool, 
        help="Use uncertainty from probabilistic posrec to derive weights"
        )

    merged_s2s_get_window_size_factor = straxen.URLConfig(
        default=5, type=int, track=False, help="Factor of the window size for the merged_s2s plugin"
    )

    def _have_data(self, field):
        return field in self.deps["peaklets"].dtype_for("peaklets").names

    def infer_dtype(self):
        peaklet_classification_dtype = self.deps["peaklet_classification"].dtype_for(
            "peaklet_classification"
        )
        peaklets_dtype = self.deps["peaklets"].dtype_for("peaklets")
        # The merged dtype is argument position dependent!
        # It must be first classification then peaklet
        # Otherwise strax will raise an error
        # when checking for the returned dtype!
        merged_s2s_dtype = strax.merged_dtype((peaklet_classification_dtype, peaklets_dtype))
        enhanced_peaklet_classification_dtype = self.deps["peaklet_classification"].dtype_for(
            "peaklet_classification"
        )
        return dict(
            merged_s2s=merged_s2s_dtype,
            enhanced_peaklet_classification=enhanced_peaklet_classification_dtype,
        )

    def setup(self):
        self.to_pe = self.gain_model
        self.gap_thresholds = np.array(self.s2_merge_gap_thresholds).T
        # Max gap and area should be set by the gap thresholds to avoid contradictions
        if np.argmax(self.gap_thresholds[1]) != 0:
            raise ValueError("The first point should be the maximum gap to allow merging")
        self.max_gap = self.gap_thresholds[1, 0]
        self.max_area = 10 ** self.gap_thresholds[0, -1]
        self.max_duration = self.s2_merge_max_duration

        self.poisson_max_k = np.ceil(
            poisson.isf(q=self.poisson_survival_ratio, mu=self.poisson_max_mu)
        ).astype(int)
        self.factorial_panel = np.array(
            [np.prod(np.arange(1, k + 1, dtype=float)) for k in range(self.poisson_max_k + 1)]
        )
        if np.any(self.factorial_panel < 0):
            raise ValueError("Factorial panel has negative values, this might because of overflow")
        x = np.linspace(-self.normal_max_sigma, self.normal_max_sigma, self.normal_n_bins)
        self.normal_panel = np.vstack([x, norm.cdf(x)])

        self.sigma = np.linspace(self.rough_min_sigma, self.rough_max_sigma, self.rough_sigma_bins)
        self.sigma_panel = np.repeat(self.sigma[None, :], self.rough_mu_bins, axis=0).flatten()

    def get_window_size(self):
        return self.merged_s2s_get_window_size_factor * (
            int(self.s2_merge_gap_thresholds[0][1]) + self.s2_merge_max_duration
        )

    def no_need(self, peaklets):
        is_s2 = peaklets["type"] == 2
        return np.sum(is_s2) <= 1 or self.max_gap < 0

    def compute(self, peaklets, lone_hits, start, end):
        # initialize enhanced_peaklet_classification
        enhanced_peaklet_classification = np.zeros(
            len(peaklets), dtype=self.dtype_for("enhanced_peaklet_classification")
        )
        # copy fields, especially type
        for d in enhanced_peaklet_classification.dtype.names:
            enhanced_peaklet_classification[d] = peaklets[d]

        if self.no_need(peaklets):
            empty_result = self.empty_result()
            empty_result["enhanced_peaklet_classification"] = enhanced_peaklet_classification
            return empty_result

        is_s2 = peaklets["type"] == 2

        # peaklets might be overwritten in the merge method
        # so do not reuse the peaklets after this line
        merged_s2s, merged = self.merge(peaklets, lone_hits, start, end)

        # mark the peaklets can be merged by time-density but not
        # by position-density as type WIDE_XYPOS_S2_TYPE
        enhanced_peaklet_classification["type"][is_s2 & ~merged] = WIDE_XYPOS_S2_TYPE

        return dict(
            merged_s2s=merged_s2s, enhanced_peaklet_classification=enhanced_peaklet_classification
        )

    def merge(self, peaklets, lone_hits, start, end):
        """Merge into S2s if the peaklets are close enough in time and position."""
        # only keep S2 peaklets for merging
        is_s2 = peaklets["type"] == 2
        peaklets = peaklets[is_s2]

        max_buffer = int(self.max_duration // strax.gcd_of_array(peaklets["dt"]))

        if not (self._have_data("data_top") and self._have_data("data_start")):
            peaklets_w_field = np.zeros(
                len(peaklets),
                dtype=strax.peak_dtype(
                    n_channels=self.n_tpc_pmts, store_data_top=True, store_data_start=True
                ),
            )
            strax.copy_to_buffer(peaklets, peaklets_w_field, "_add_data_top_or_start_field")
            del peaklets
            peaklets = peaklets_w_field

        start_merge_at, end_merge_at, _merged = self.get_merge_instructions(
            np.copy(peaklets),
            start,
            end,
            self.max_gap,
            self.max_duration,
            self.sigma,
            self.rough_seg,
            self.sigma_seg,
            self.rough_mu_bins,
            self.poisson_max_mu,
            self.poisson_survival_ratio,
            self.normal_panel,
            self.factorial_panel,
            self.sigma_panel,
            self.maxexp,
            self.n_top_pmts,
            self.p_threshold,
            self.dr_threshold,
            self.use_bayesian_merging,
            self.gap_thresholds,
        )

        if "data_top" not in peaklets.dtype.names or "data_start" not in peaklets.dtype.names:
            raise ValueError("data_top or data_start is not in the peaklets dtype")

        # have to redo the merging to prevent numerical instability
        merged_s2s = self.merge_peaklets(
            peaklets, start_merge_at, end_merge_at, _merged, max_buffer
        )
        if np.max((strax.endtime(merged_s2s) - merged_s2s["time"])) > self.max_duration:
            raise ValueError("Merged S2 is too long")

        # Updated time and length of lone_hits and sort again:
        # this is an OverlapWindowPlugin, some lone_hits will be reused in the next iteration
        # so do not overwirte the lone_hits
        lh = np.copy(lone_hits)
        del lone_hits
        lh_time_shift = (lh["left"] - lh["left_integration"]) * lh["dt"]
        lh["time"] = lh["time"] - lh_time_shift
        lh["length"] = lh["right_integration"] - lh["left_integration"]
        lh = strax.sort_by_time(lh)

        _store_data_top = "data_top" in self.dtype_for("merged_s2s").names
        _store_data_start = "data_start" in self.dtype_for("merged_s2s").names
        strax.add_lone_hits(
            merged_s2s,
            lh,
            self.to_pe,
            n_top_channels=self.n_top_pmts,
            store_data_top=_store_data_top,
            store_data_start=_store_data_start,
        )

        strax.compute_properties(merged_s2s, n_top_channels=self.n_top_pmts)

        # remove position fields
        merged_s2s = drop_data_field(
            merged_s2s, self.dtype_for("merged_s2s"), "_drop_data_field_merged_s2s"
        )

        # make sure merged has same length as peaklets
        merged = np.zeros(len(is_s2), dtype=bool)
        merged[is_s2] = _merged

        return merged_s2s, merged

    @staticmethod
    def get_left_right(peaklets):
        """Get the left and right boundaries of the peaklets."""
        # The gap is defined as the 90% to 10% area decile distance of the adjacent peaks
        left = (peaklets["area_decile_from_midpoint"][1] + peaklets["median_time"]).astype(int)
        right = (peaklets["area_decile_from_midpoint"][9] + peaklets["median_time"]).astype(int)
        return left, right

    @staticmethod
    def get_gap(y, x):
        """Get the gap between two peaklets."""
        y_left, y_right = MergedS2s.get_left_right(y)
        x_left, x_right = MergedS2s.get_left_right(x)
        dt = x["time"] - y["time"]
        boundaries = np.sort([y_left, y_right, x_left + dt, x_right + dt])
        this_gap = boundaries[2] - boundaries[1]
        return this_gap

    @staticmethod
    def get_duration(y, x):
        """Get the duration of the merged peaklets."""
        return max(strax.endtime(y), strax.endtime(x)) - min(y["time"], x["time"])

    @staticmethod
    def get_merge_instructions(
        peaks,
        start,
        end,
        max_gap,
        max_duration,
        sigma,
        rough_seg,
        sigma_seg,
        rough_mu_bins,
        poisson_max_mu,
        poisson_survival_ratio,
        normal_panel,
        factorial_panel,
        sigma_panel,
        maxexp,
        n_top_pmts,
        p_threshold,
        dr_threshold,
        bayesian=True,
        gap_thresholds=None,
        diagnosing=False,
        disable=True,
    ):
        """Find the group of peaklets to merge.

        There are two ways to merge peaklets:
        1. Bayesian merging: merge peaklets based on the p-value of the time-density merging
        2. Normal merging: merge peaklets based on the gap between the peaklets

        """
        max_buffer = int(max_duration // strax.gcd_of_array(peaks["dt"]))

        n_peaks = len(peaks)
        if n_peaks == 0:
            raise ValueError("No peaklets to merge")

        start_index = np.arange(n_peaks)
        # exclusive end index
        end_index = np.arange(n_peaks) + 1

        # mask to help keep track of the peaklets that have been examined
        unexamined = np.full(n_peaks, True)
        # mask to help keep track of the peaklets that should not be merged because
        # of too high standard deviation from the main cluster in (x, y) of the peaklets
        merged = np.full(n_peaks, True)

        # approximation of the integration boundaries
        if np.any(peaks["time"][1:] < strax.endtime(peaks)[:-1]):
            raise ValueError("Peaks not disjoint, why?")
        core_bounds = (peaks["time"][1:] + strax.endtime(peaks)[:-1]) // 2
        # here the constraint on boundaries is also to make sure get_window_size covers the gaps
        left_bounds = np.maximum(np.hstack([start, core_bounds]), peaks["time"] - int(max_gap / 2))
        right_bounds = np.minimum(
            np.hstack([core_bounds, end]), strax.endtime(peaks) + int(max_gap / 2)
        )

        # (x, y) positions of the peaklets
        positions = np.vstack(
            [peaks[f"x_{DEFAULT_POSREC_ALGO}"], peaks[f"y_{DEFAULT_POSREC_ALGO}"]]
        ).T

        # weights of the peaklets when calculating the weighted mean deviation in (x, y)
        area = np.copy(peaks["area"])
        area_top = area * peaks["area_fraction_top"]

        if diagnosing:
            merged_area = []
            p_values = []

        argsort = np.argsort(peaks["area"], kind="mergesort")
        for i in tqdm(argsort[::-1], disable=disable):
            if not unexamined[i]:
                continue
            p = [1.0, 1.0]
            # in the while loops, the peaklets will be merged until the p-value
            # is smaller than the threshold or the peaklets can not be merged anymore
            # i will NOT be updated in the while loop
            while max(p) >= p_threshold and unexamined[i]:
                indices = []
                # please mind here that the definition of gaps should
                # be consistent with in the merging algorithm
                # and the merged peak should not be longer than the max duration
                if (
                    start_index[i] - 1 >= 0
                    and area[start_index[i] - 1] > 0
                    and unexamined[start_index[i] - 1]
                    and MergedS2s.get_gap(peaks[i], peaks[start_index[i] - 1]) < max_gap
                    and MergedS2s.get_duration(peaks[i], peaks[start_index[i] - 1]) <= max_duration
                ):
                    indices.append(start_index[i] - 1)
                else:
                    indices.append(None)
                if (
                    end_index[i] < n_peaks
                    and area[end_index[i]] > 0
                    and unexamined[end_index[i]]
                    and MergedS2s.get_gap(peaks[i], peaks[end_index[i]]) < max_gap
                    and MergedS2s.get_duration(peaks[i], peaks[end_index[i]]) <= max_duration
                ):
                    indices.append(end_index[i])
                else:
                    indices.append(None)
                p = []
                # get p-values
                for j in indices:
                    if j is None:
                        p.append(-1.0)
                        continue
                    if bayesian:
                        p_ = get_p_value(
                            peaks[i],
                            peaks[j],
                            left_bounds[i],
                            right_bounds[i],
                            left_bounds[j],
                            right_bounds[j],
                            sigma,
                            rough_seg,
                            sigma_seg,
                            rough_mu_bins,
                            poisson_max_mu,
                            poisson_survival_ratio,
                            normal_panel,
                            factorial_panel,
                            sigma_panel,
                            maxexp,
                        )
                    else:
                        # this is kept for diagnosing and merged_s2s_he
                        this_gap = MergedS2s.get_gap(peaks[i], peaks[j])
                        gap_threshold = merge_s2_threshold(
                            np.log10(peaks["area"][i] + peaks["area"][j]),
                            gap_thresholds,
                        )
                        # gap of 90-10% area decile distance should not be larger than the threshold
                        if this_gap < gap_threshold:
                            p_ = 1.0
                        else:
                            p_ = -1.0
                    p.append(p_)
                examined = slice(start_index[i], end_index[i])
                if diagnosing:
                    merged_area.append(area[examined][merged[examined]].sum())
                    p_values.append(max(p))

                if max(p) < p_threshold:
                    # this will not allow merging of the already examined peaklets
                    unexamined[examined] = False
                else:
                    # slice indicating the direction of merging
                    if p[0] >= p_threshold and p[0] > p[1]:
                        direction = slice(indices[0], indices[0] + 2)
                        index = indices[0]
                    elif p[1] >= p_threshold and p[1] >= p[0]:
                        direction = slice(indices[1] - 1, indices[1] + 1)
                        index = indices[1]
                    else:
                        raise RuntimeError("Can not decide to merge or not")
                    # slice indicating the peaklets to be merged
                    start_idx = start_index[direction.start]
                    end_idx = end_index[direction.stop - 1]
                    merging = slice(start_idx, end_idx)

                    # calculate weighted averaged deviation of peaklets from the main cluster
                    if use_uncertainty_weights:
                        contour_areas = polygon_area(
                            peaks[f'position_contour_{DEFAULT_POSREC_ALGO}'][merging][merged[merging]]
                            )
                        weights = np.nan_to_num(
                            1/contour_areas, 
                            nan=np.finfo('float32').tiny
                            )
                    else:
                        weights = area_top[merging][merged[merging]]
                    dr_avg = weighted_averaged_dr(
                        positions[merging, 0][merged[merging]],
                        positions[merging, 1][merged[merging]],
                        weights,
                    )
                    # do we really merge the peaklets?
                    merge = dr_avg < dr_threshold

                    if merge:
                        merged_peak = strax.merge_peaks(
                            peaks[direction],
                            [0],
                            [2],
                            max_buffer=max_buffer,
                        )
                        # this step is necessary to update the properties of the merged peak
                        # but the properties will be calculated again after merging
                        # to keep numerical stability
                        strax.compute_properties(merged_peak, n_top_channels=n_top_pmts)
                        merged_peak = merged_peak[0]
                        # check if the merged peak is too long
                        if strax.endtime(merged_peak) - merged_peak["time"] > max_duration:
                            raise ValueError("Merged S2 is too long")
                    else:
                        merged[index] = False
                        merged_peak = peaks[i]

                    # update merging peaks and boundaries
                    # update ALL peaks in the slice, this is only temporarily needed
                    # because we can not foresee in which order the peaks will be merged
                    peaks[merging] = merged_peak
                    start_index[merging] = start_index[merging.start]
                    end_index[merging] = end_index[merging.stop - 1]
                    left_bounds[merging] = left_bounds[merging.start]
                    right_bounds[merging] = right_bounds[merging.stop - 1]
        if np.any(np.diff(start_index) < 0) or np.any(np.diff(end_index) < 0):
            raise ValueError("Indices are not sorted!")
        n_peaklets = end_index - start_index
        need_merging = n_peaklets > 1
        start_index = np.unique(start_index[need_merging])
        end_index = np.unique(end_index[need_merging])
        if diagnosing:
            return start_index, end_index, merged, merged_area, p_values
        return start_index, end_index, merged

    @staticmethod
    def merge_peaklets(
        peaklets, start_merge_at, end_merge_at, merged, max_buffer, merged_all=False
    ):
        if merged_all:
            # execute earlier to prevent peaklets from being overwritten
            # mark the peaklets can be merged by time-density but not
            # by position-density as type WIDE_XYPOS_S2_TYPE
            _merged_s2s = np.copy(peaklets[~merged])
            _merged_s2s["type"] = WIDE_XYPOS_S2_TYPE
        # mark the peaklets can be merged by time-density and position-density as type 2
        merged_s2s = strax.merge_peaks(
            peaklets,
            start_merge_at,
            end_merge_at,
            merged=merged,
            max_buffer=max_buffer,
        )
        merged_s2s["type"] = 2
        if merged_all:
            merged_s2s = strax.sort_by_time(np.concatenate([_merged_s2s, merged_s2s]))
        return merged_s2s


@numba.njit(cache=True, nogil=True)
def get_p_value(
    y,
    x,
    ly,
    ry,
    lx,
    rx,
    sigma,
    rough_seg,
    sigma_seg,
    rough_mu_bins,
    poisson_max_mu,
    poisson_survival_ratio,
    normal_panel,
    factorial_panel,
    sigma_panel,
    maxexp,
):
    y_time = y["time"]
    x_time = x["time"]
    y_median_time = y["median_time"]
    x_median_time = x["median_time"]
    y_data = y["data"]
    y_dt = y["dt"]
    t0 = y["time"]

    # better constraint the mu in boundaries of waveform
    # because the number of bins is not high enough comprising for computation efficiency
    mu = rough_mu(y, x, rough_mu_bins)

    y_integrated = normal_cdf(((ry - t0) - mu[:, None]) / sigma, normal_panel)
    y_integrated -= normal_cdf(((ly - t0) - mu[:, None]) / sigma, normal_panel)
    x_integrated = normal_cdf(((rx - t0) - mu[:, None]) / sigma, normal_panel)
    x_integrated -= normal_cdf(((lx - t0) - mu[:, None]) / sigma, normal_panel)
    # keep numerical stability
    eps = 2.220446049250313e-16  # np.finfo(float).eps
    y_integrated += eps
    x_integrated += eps

    pmf = get_posterior(
        mu,
        sigma,
        y_dt,
        y_data,
        y_integrated,
        y_time,
        y_median_time,
        x_time,
        x_median_time,
        rough_seg,
        maxexp,
    ).flatten()

    non_zero = pmf > 0
    if not np.any(non_zero):
        return 0.0
    # the mask non_zero here is also implicit flatten
    ps = p_values(
        sigma_panel[non_zero],
        y["area"],
        y_integrated.flatten()[non_zero],
        x["area"],
        x_integrated.flatten()[non_zero],
        rough_seg,
        sigma_seg,
        poisson_max_mu,
        poisson_survival_ratio,
        normal_panel,
        factorial_panel,
    )

    p = np.sum(ps * pmf[non_zero])
    if np.isnan(p):
        raise RuntimeError("p-value is NaN")
    return p


@numba.njit(cache=True, nogil=True)
def merge_s2_threshold(log_area, gap_thresholds):
    """Return gap threshold for log_area of the merged S2 with linear interpolation given the points
    in gap_thresholds.

    :param log_area: Log 10 area of the merged S2
    :param gap_thresholds: tuple (n, 2) of fix points for interpolation.

    """
    if log_area < gap_thresholds[0, 0]:
        return gap_thresholds[1, 0]
    if log_area > gap_thresholds[0, -1]:
        return gap_thresholds[1, -1]
    return np.interp(log_area, gap_thresholds[0], gap_thresholds[1])


@numba.njit(cache=True, nogil=True)
def rough_mu(y, x, rough_mu_bins):
    """Get rough bins for mu as the integration space."""
    mu = np.linspace(
        min(y["time"], x["time"]) - y["time"],
        max(strax.endtime(y), strax.endtime(x)) - y["time"],
        rough_mu_bins,
    )
    return mu


@numba.njit(cache=True, nogil=True)
def poisson_pmf(k, mu, factorial_panel):
    """Probability mass function of Poisson distribution, numba decorated."""
    return mu**k / np.exp(mu) / factorial_panel[k]


@numba.njit(cache=True, nogil=True)
def normal_cdf(x, normal_panel):
    """Cumulative density function of standard normal distribution, numba decorated."""
    return np.interp(x, normal_panel[0], normal_panel[1])


@numba.njit(cache=True, nogil=True)
def get_posterior(
    mu,
    sigma,
    y_dt,
    y_data,
    y_integrated,
    y_time,
    y_median_time,
    x_time,
    x_median_time,
    rough_seg,
    maxexp,
):
    """Get the posterior of (mu, sigma) based on the waveform y."""
    # production of normal PDF
    y_t = np.arange(y_data.size) * y_dt
    sigma2 = 2 * sigma**2
    log_pdf = -((y_t - mu[:, None]) ** 2 * y_data).sum(axis=1)[:, None] / (rough_seg * sigma2)
    log_sigma = np.log(sigma)
    y_nse = y_data.sum() / rough_seg
    log_pdf -= log_sigma * y_nse
    # account for the bounded sampling of y
    log_pdf -= np.log(y_integrated) * y_nse
    # add log prior
    # dt is median between median time of two peaklets
    dt = np.abs((y_time - x_time) + (y_median_time - x_median_time))
    log_pdf -= dt**2 / 4 / sigma2 + 2 * log_sigma
    # to prevent numerical overflow
    log_pdf -= log_pdf.max()
    log_pdf += maxexp
    pdf = np.exp(log_pdf)
    pmf = pdf / pdf.sum()
    if np.any(np.isnan(pmf)):
        raise RuntimeError("pmf is NaN")
    return pmf


@numba.njit(cache=True, nogil=True)
def p_values(
    sigma,
    y_area,
    y_integrated,
    x_area,
    x_integrated,
    rough_seg,
    sigma_seg,
    poisson_max_mu,
    poisson_survival_ratio,
    normal_panel,
    factorial_panel,
):
    """The p-value if observe the x_area given the y_area, numba decorated."""
    # distribution of allowed area
    expected_nse = x_integrated * y_area / y_integrated / rough_seg
    ps = []
    for mu in expected_nse:
        # the number of electron follows Poisson distribution
        if mu > poisson_max_mu:
            ps.append(1 - normal_cdf((x_area / rough_seg - mu) / np.sqrt(mu), normal_panel))
        else:
            cdf = poisson_pmf(0, mu, factorial_panel)
            k = 1
            p = 0
            while cdf < 1 - poisson_survival_ratio:
                pmf = poisson_pmf(k, mu, factorial_panel)
                cdf += pmf
                p += pmf * (
                    1 - normal_cdf((x_area - k * rough_seg) / (k**0.5 * sigma_seg), normal_panel)
                )
                k += 1
            ps.append(p)
    ps = np.array(ps)
    return ps


@numba.njit(cache=True, nogil=True)
def weighted_averaged_dr(x, y, weights):
    """Weighted average deviation from weighted average (x, y)"""
    mask = weights > 0
    mask &= ~np.isnan(x)
    mask &= ~np.isnan(x)
    if not np.any(mask):
        return np.nan
    x_avg = np.average(x[mask], weights=weights[mask])
    y_avg = np.average(y[mask], weights=weights[mask])
    dr = np.sqrt((x - x_avg) ** 2 + (y - y_avg) ** 2)
    dr_avg = np.average(dr[mask], weights=weights[mask])
    return dr_avg

@numba.jit(cache=True)
def polygon_area(polygon):
    """
    Calculate and return the area of a polygon.

    The input is a 3D numpy array where the first dimension represents individual polygons,
    the second dimension represents vertices of the polygon, 
    and the third dimension represents x and y coordinates of each vertex.
    """
    x = polygon[..., 0]
    y = polygon[..., 1]
    result = np.zeros(polygon.shape[0], dtype='float32')
    for i in range(x.shape[-1]):
        result +=  (x[..., i]*y[..., i-1]) - (y[..., i]*x[..., i-1])
    return 0.5 * np.abs(result)
