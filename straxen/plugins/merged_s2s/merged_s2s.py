from typing import Tuple, Dict, Union
import numpy as np
from tqdm import tqdm
from scipy.stats import norm, poisson
import numba
import strax
import straxen
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO, FAR_XYPOS_S2_TYPE, WIDE_XYPOS_S2_TYPE
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
            (2.18, 2.40e04),
            (2.51, 1.96e04),
            (2.84, 1.80e04),
            (3.18, 1.68e04),
            (3.51, 1.86e04),
            (3.84, 1.98e04),
            (4.18, 1.66e04),
            (4.51, 1.21e04),
        ),
        infer_type=False,
        help=(
            "Points to define maximum separation between peaklets to allow "
            "merging [ns] depending on log10 area of the merged peak\n"
            "where the gap size of the first point is the maximum gap to allow merging."
            "The format is ((log10(area), max_gap), (..., ...), (..., ...))"
        ),
    )

    s2_merge_p_thresholds = straxen.URLConfig(
        default=(
            (4.18, 9.89e-04),
            (4.51, 7.15e-03),
            (4.84, 1.60e-01),
        ),
        infer_type=False,
        help=(
            "Points to define minimum p-value of merging proposal "
            "depending on log10 area of the merged peak\n"
            "The format is ((log10(area), p-value), (..., ...), (..., ...))"
        ),
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

    s2_merge_unmerged_thresholds = straxen.URLConfig(
        default=(1, 49.5, 0.01),
        type=tuple,
        help=(
            "Max (number, fraction of unmerged, total area of unmerged) "
            "of type 20 peaklets inside a peak. "
            "The number of type 20 peaklets should not be larger than the number threshold. "
            "The area of type 20 peaklets should not be larger than the both area thresholds. "
            "The fraction threshold is important when S2 is large, "
            "while the total area threshold is important when S2 is small."
        ),
    )

    merge_lone_hits = straxen.URLConfig(
        default=True,
        type=bool,
        help="Merge lone hits into merged S2s",
    )

    merge_s0 = straxen.URLConfig(
        default=True,
        type=bool,
        help="Merge S0s into merged S2s",
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

    use_bayesian_merging = straxen.URLConfig(default=True, type=bool, help="Use Bayesian merging")

    rm_sparse_xy = straxen.URLConfig(
        default=True, type=bool, help="Remove peaklets that are too far away in (x, y)"
    )

    use_uncertainty_weights = straxen.URLConfig(
        default=True, type=bool, help="Use uncertainty from probabilistic posrec to derive weights"
    )

    p_value_prioritized = straxen.URLConfig(
        default=False,
        type=bool,
        help="Whether to prioritize p-value over area when testing the proposal",
    )

    merged_s2s_get_window_size_factor = straxen.URLConfig(
        default=5, type=int, track=False, help="Factor of the window size for the merged_s2s plugin"
    )

    disable_progress_bar = straxen.URLConfig(
        default=True, type=bool, track=False, help="Whether to disable the progress bar"
    )

    copied_dtype = np.dtype(
        [
            ("time", np.int64),
            ("endtime", np.int64),
            ("area", np.float32),
            ("median_time", np.float32),
            ("area_decile_from_midpoint", np.float32, (11,)),
        ]
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
        self.p_thresholds = np.array(self.s2_merge_p_thresholds).T
        self.dr_thresholds = np.array(self.s2_merge_dr_thresholds).T
        # Max gap and area should be set by the gap thresholds to avoid contradictions
        if np.argmax(self.gap_thresholds[1]) != 0:
            raise ValueError("The first point should be the maximum gap to allow merging")
        if self.p_thresholds[1].max() > 1 or self.p_thresholds[1].min() < 0:
            raise ValueError("P-value should be in the range of [0, 1]")
        self.max_gap = self.gap_thresholds[1, 0]
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

        self.unmerged_thresholds = self.s2_merge_unmerged_thresholds

    def get_window_size(self):
        return self.merged_s2s_get_window_size_factor * (
            int(self.s2_merge_gap_thresholds[0][1]) + self.s2_merge_max_duration
        )

    def no_merging(self, peaklets):
        is_s2 = peaklets["type"] == 2
        return np.sum(is_s2) <= 1 or self.max_gap < 0

    def compute(self, peaklets, lone_hits, start, end):
        if self.use_uncertainty_weights:
            name = f"position_contour_{self.default_reconstruction_algorithm}"
            if name not in peaklets.dtype.names:
                raise ValueError(f"{name} is not in the input peaklets dtype")

        # initialize enhanced_peaklet_classification
        enhanced_peaklet_classification = np.zeros(
            len(peaklets), dtype=self.dtype_for("enhanced_peaklet_classification")
        )
        # copy fields, especially type
        for d in enhanced_peaklet_classification.dtype.names:
            enhanced_peaklet_classification[d] = peaklets[d]

        if self.no_merging(peaklets):
            empty_result = self.empty_result()
            empty_result["enhanced_peaklet_classification"] = enhanced_peaklet_classification
            return empty_result

        # make sure the peaklets are not overwritten
        peaklets.flags.writeable = False

        is_s2 = peaklets["type"] == 2

        # peaklets might be overwritten in the merge method
        # so do not reuse the peaklets after this line
        merged_s2s, merged = self.merge(peaklets, lone_hits, start, end)

        # mark the peaklets can be merged by time-density but not
        # by position-density as type FAR_XYPOS_S2_TYPE
        enhanced_peaklet_classification["type"][is_s2 & ~merged] = FAR_XYPOS_S2_TYPE

        return dict(
            merged_s2s=merged_s2s, enhanced_peaklet_classification=enhanced_peaklet_classification
        )

    def merge(self, _peaklets, lone_hits, start, end):
        """Merge into S2s if the peaklets are close enough in time and position."""
        # this is an OverlapWindowPlugin, some peaklets will be reused in the next iteration
        # use _peaklets here to prevent peaklets from being overwritten
        if np.any(_peaklets["time"][1:] < strax.endtime(_peaklets)[:-1]):
            raise ValueError("Peaklets not disjoint, why?")

        # only keep S2 peaklets for merging
        is_s2 = _peaklets["type"] == 2

        if not (self._have_data("data_top") and self._have_data("data_start")):
            peaklets = np.zeros(
                is_s2.sum(),
                dtype=strax.peak_dtype(
                    n_channels=self.n_tpc_pmts, store_data_top=True, store_data_start=True
                ),
            )
            strax.copy_to_buffer(_peaklets[is_s2], peaklets, "_add_data_top_or_start_field")
        else:
            peaklets = _peaklets[is_s2]

        # make sure the peaklets are not overwritten
        peaklets.flags.writeable = False

        max_buffer = int(self.max_duration // strax.gcd_of_array(peaklets["dt"]))

        start_merge_at, end_merge_at, _merged = self.get_merge_instructions(
            peaklets,
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
            self.p_thresholds,
            self.dr_thresholds,
            self.default_reconstruction_algorithm,
            self.use_bayesian_merging,
            self.rm_sparse_xy,
            self.use_uncertainty_weights,
            self.p_value_prioritized,
            self.gap_thresholds,
            disable=self.disable_progress_bar,
        )

        if "data_top" not in peaklets.dtype.names or "data_start" not in peaklets.dtype.names:
            raise ValueError("data_top or data_start is not in the peaklets dtype")

        # have to redo the merging to prevent numerical instability
        is_s0 = _peaklets["type"] == 0
        if self.merge_s0 and len(start_merge_at) and is_s0.sum() > 0:
            # build the time interval of merged_s2s, even though they are not merged yet
            endtime = strax.endtime(peaklets)
            merged_s2s_window = np.zeros(len(start_merge_at), dtype=strax.time_fields)
            for i in range(len(start_merge_at)):
                sl = slice(start_merge_at[i], end_merge_at[i])
                merged_s2s_window["time"][i] = peaklets["time"][sl][_merged[sl]][0]
                merged_s2s_window["endtime"][i] = endtime[sl][_merged[sl]][-1]
            # the S0s that should be merged should fully be contained
            merged_s0s = strax.split_by_containment(_peaklets[is_s0], merged_s2s_window)
            # offsets of indices
            increments = np.array([len(m) for m in merged_s0s], dtype=int)
            offsets = np.hstack([0, np.cumsum(increments)])
            _start_merge_at = start_merge_at + offsets[:-1]
            _end_merge_at = end_merge_at + offsets[1:]
            if np.min(_end_merge_at - _start_merge_at) < 2:
                raise ValueError("You are merging nothing!")
            # prepare for peaklets including S0s
            __merged = np.hstack([_merged, np.full(increments.sum(), True)])
            _peaklets = np.hstack([peaklets, np.hstack(merged_s0s)])
            argsort = strax.stable_argsort(_peaklets["time"])
            merged_s2s = self.merge_peaklets(
                _peaklets[argsort],
                _start_merge_at,
                _end_merge_at,
                __merged[argsort],
                max_buffer,
                max_unmerged=self.unmerged_thresholds,
            )
        else:
            merged_s2s = self.merge_peaklets(
                peaklets,
                start_merge_at,
                end_merge_at,
                _merged,
                max_buffer,
                max_unmerged=self.unmerged_thresholds,
            )

        if (
            len(merged_s2s)
            and np.max((strax.endtime(merged_s2s) - merged_s2s["time"])) > self.max_duration
        ):
            raise ValueError("Merged S2 is too long")

        if self.merge_lone_hits:
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
    def get_left_right(peaklet):
        """Get the left and right boundaries of the peaklet."""
        # The gap is defined as the 90% to 10% area decile distance of the adjacent peaks
        left = (peaklet["area_decile_from_midpoint"][1] + peaklet["median_time"]).astype(int)
        right = (peaklet["area_decile_from_midpoint"][9] + peaklet["median_time"]).astype(int)
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
    @numba.njit(cache=True, nogil=True)
    def get_duration(y, x):
        """Get the duration of the merged peaklets."""
        return max(y["endtime"], x["endtime"]) - min(y["time"], x["time"])

    @staticmethod
    @numba.njit(cache=True, nogil=True)
    def decile_interp(
        x,
        y,
        panel=np.linspace(0, 1, 11),
    ):
        area = x["area"] + y["area"]
        xf = x["area"] / area
        yf = y["area"] / area
        xp = np.concatenate((panel * xf, panel * yf + xf))
        yp = np.concatenate(
            (
                (x["area_decile_from_midpoint"] + x["median_time"]),
                (y["time"] - x["time"]) + (y["area_decile_from_midpoint"] + y["median_time"]),
            )
        )
        decile = np.interp(panel, xp, yp)
        return decile[5], decile - decile[5]

    @staticmethod
    def merge_two_peaks(x, y):
        """Merge two peaklets in order."""
        z = np.array(0, dtype=x.dtype)
        z["time"] = x["time"]
        z["endtime"] = y["endtime"]
        z["area"] = x["area"] + y["area"]
        z["median_time"], z["area_decile_from_midpoint"] = MergedS2s.decile_interp(x, y)
        return z

    @staticmethod
    def get_merge_instructions(
        _peaks,
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
        p_thresholds,
        dr_thresholds,
        posrec_algo,
        bayesian=True,
        sparse_xy=True,
        uncertainty_weights=True,
        p_value_prioritized=False,
        gap_thresholds=None,
        diagnosing=False,
        disable=True,
    ):
        """Find the group of peaklets to merge.

        There are two ways to merge peaklets:
        1. Bayesian merging: merge peaklets based on the p-value of the time-density merging
        2. Normal merging: merge peaklets based on the gap between the peaklets

        """

        # (x, y) positions of the peaklets
        positions = np.vstack([_peaks[f"x_{posrec_algo}"], _peaks[f"y_{posrec_algo}"]]).T
        if uncertainty_weights:
            contour_area = _peaks[f"position_contour_area_{posrec_algo}"]
        # weights of the peaklets when calculating the weighted mean deviation in (x, y)
        area = _peaks["area"]
        area_top = area * _peaks["area_fraction_top"]

        peaks = np.zeros(len(_peaks), dtype=MergedS2s.copied_dtype)
        for name in peaks.dtype.names:
            if name == "endtime":
                peaks[name] = strax.endtime(_peaks)
            else:
                peaks[name] = _peaks[name]
        peaks_copy = np.copy(peaks)
        peaks_copy.flags.writeable = False

        n_peaks = len(peaks)
        if n_peaks == 0:
            raise ValueError("No peaklets to merge")

        start_index = np.arange(n_peaks)
        # exclusive end index
        end_index = np.arange(n_peaks) + 1
        # keep a copy of the original indices
        start_index_copy = np.copy(start_index)
        start_index_copy.flags.writeable = False
        end_index_copy = np.copy(end_index)
        end_index_copy.flags.writeable = False

        # mask to help keep track of the peaklets that have been examined
        unexamined = np.full(n_peaks, True)
        # mask to help keep track of the peaklets that should not be merged because
        # of too high standard deviation from the main cluster in (x, y) of the peaklets
        merged = np.full(n_peaks, True)

        # approximation of the integration boundaries
        core_bounds = (peaks["time"][1:] + peaks["endtime"][:-1]) // 2
        # here the constraint on boundaries is also to make sure get_window_size covers the gaps
        left_bounds = np.maximum(np.hstack([start, core_bounds]), peaks["time"] - int(max_gap / 2))
        right_bounds = np.minimum(
            np.hstack([core_bounds, end]), peaks["endtime"] + int(max_gap / 2)
        )
        # keep a copy of the original boundaries
        left_bounds_copy = np.copy(left_bounds)
        left_bounds_copy.flags.writeable = False
        right_bounds_copy = np.copy(right_bounds)
        right_bounds_copy.flags.writeable = False

        if diagnosing:
            merged_area = []
            p_values = []
            dr_avgs = []

        def gap_small_enough(y, x):
            log_area = np.log10(y["area"] + x["area"])
            this_gap = MergedS2s.get_gap(y, x)
            gap_threshold = thresholds_interpolation(log_area, gap_thresholds)
            # gap of 90-10% area decile distance should not be larger than the threshold
            return this_gap < gap_threshold

        def can_merge_left(i, peaks):
            # whether peaklet i (from _peak) can be merged to its left unexamined peaklet
            flag = (
                i - 1 >= 0
                and unexamined[i - 1]
                and MergedS2s.get_gap(peaks[i], peaks[i - 1]) < max_gap
                and MergedS2s.get_duration(peaks[i], peaks[i - 1]) <= max_duration
            )
            if not flag:
                return False
            return gap_small_enough(peaks[i], peaks[i - 1])

        def can_merge_right(i, peaks):
            # whether peaklet i (from _peak) can be merged to its right unexamined peaklet
            flag = (
                i < n_peaks
                and unexamined[i]
                and MergedS2s.get_gap(peaks[i - 1], peaks[i]) < max_gap
                and MergedS2s.get_duration(peaks[i - 1], peaks[i]) <= max_duration
            )
            if not flag:
                return False
            return gap_small_enough(peaks[i - 1], peaks[i])

        argsort = strax.stable_argsort(peaks["area"])
        for i in tqdm(argsort[::-1], disable=disable):
            if not unexamined[i]:
                continue
            p = np.array([1.0, 1.0])
            p_threshold = np.array([0.0, 0.0])
            left = True
            dr_avg = 0.0
            # in the while loops, the peaklets will be merged until the p-value
            # is smaller than the threshold or the peaklets can not be merged anymore
            # i will NOT be updated in the while loop
            while np.any(p >= p_threshold) and unexamined[i]:
                indices = []
                # please mind here that the definition of gaps should
                # be consistent with in the merging algorithm
                # and the merged peak should not be longer than the max duration
                if can_merge_left(start_index[i], peaks):
                    indices.append(start_index[i] - 1)
                else:
                    indices.append(None)
                if can_merge_right(end_index[i], peaks):
                    indices.append(end_index[i])
                else:
                    indices.append(None)
                p = []
                p_threshold = []
                _area = []
                # get p-values
                for j in indices:
                    if j is None:
                        p.append(-1.0)
                        p_threshold.append(1.0)
                        _area.append(-1.0)
                        continue
                    # if area is non-positive, "merge" in time
                    # but still skip it later in (x, y) because its weight is nan
                    if not peaks["area"][j] > 0:
                        p.append(1.0)
                        p_threshold.append(-1.0)
                        _area.append(peaks["area"][j])
                        continue
                    log_area = np.log10(peaks["area"][i] + peaks["area"][j])
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
                        p_threshold_ = thresholds_interpolation(
                            log_area,
                            p_thresholds,
                        )
                    else:
                        # this is kept for diagnosing and merged_s2s_he
                        p_ = 1.0
                        p_threshold_ = 0.0
                    p.append(p_)
                    p_threshold.append(p_threshold_)
                    _area.append(peaks["area"][j])
                p = np.array(p)
                p_threshold = np.array(p_threshold)
                _area = np.array(_area)
                examined = slice(start_index[i], end_index[i])

                if diagnosing:
                    merged_area.append(area[examined][merged[examined]].sum())

                if np.all(p < p_threshold):
                    # this will not allow merging of the already examined peaklets
                    unexamined[examined] = False
                else:
                    # whether test the proposal with larger area first
                    if p_value_prioritized:
                        left = p[0] > p[1]
                    else:
                        left = _area[0] > _area[1]
                    # if test left first
                    if left:
                        if p[0] >= p_threshold[0]:
                            left = True
                        else:
                            assert p[1] >= p_threshold[1]
                            left = False
                    else:
                        if p[1] >= p_threshold[1]:
                            left = False
                        else:
                            assert p[0] >= p_threshold[0]
                            left = True
                    # slice indicating the direction of merging
                    if left:
                        direction = [indices[0], indices[0] + 1]
                        index = indices[0]
                    else:
                        direction = [indices[1] - 1, indices[1]]
                        index = indices[1]
                    # slice indicating the peaklets to be merged
                    start_idx = start_index[direction[0]]
                    end_idx = end_index[direction[1]]
                    merging = slice(start_idx, end_idx)

                    if sparse_xy:
                        # calculate weighted averaged deviation of peaklets from the main cluster
                        if uncertainty_weights:
                            weights = 1 / contour_area[merging][merged[merging]]
                        else:
                            weights = area_top[merging][merged[merging]]

                        dr_avg = weighted_averaged_dr(
                            positions[merging, 0][merged[merging]],
                            positions[merging, 1][merged[merging]],
                            weights,
                        )
                        # do we really merge the peaklets?
                        dr_threshold_ = thresholds_interpolation(
                            np.log10(area_top[merging][merged[merging]].sum()),
                            dr_thresholds,
                        )
                        merge = dr_avg < dr_threshold_
                    else:
                        merge = True

                    if merge:
                        merged_peak = MergedS2s.merge_two_peaks(
                            peaks[direction[0]], peaks[direction[1]]
                        )
                        # check if the merged peak is too long
                        if merged_peak["endtime"] - merged_peak["time"] > max_duration:
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

                if diagnosing:
                    if left:
                        p_values.append(p[0])
                    else:
                        p_values.append(p[1])
                    if sparse_xy:
                        dr_avgs.append(dr_avg)
                    else:
                        dr_avgs.append(0.0)

        if np.any(np.diff(start_index) < 0) or np.any(np.diff(end_index) < 0):
            raise ValueError("Indices are not sorted!")
        n_peaklets = end_index - start_index
        need_merging = n_peaklets > 1
        start_index = np.unique(start_index[need_merging])
        end_index = np.unique(end_index[need_merging])
        if diagnosing:
            return start_index, end_index, merged, merged_area, p_values, dr_avgs
        return start_index, end_index, merged

    @staticmethod
    def merge_peaklets(
        peaklets,
        start_merge_at,
        end_merge_at,
        merged,
        max_buffer=int(1e5),
        max_unmerged=None,
        merged_all=False,
    ):
        if merged_all:
            # execute earlier to prevent peaklets from being overwritten
            # mark the peaklets can be merged by time-density but not
            # by position-density as type FAR_XYPOS_S2_TYPE
            _merged_s2s = np.copy(peaklets[~merged])
            _merged_s2s["type"] = FAR_XYPOS_S2_TYPE
        # mark the peaklets can be merged by time-density and position-density as type 2
        merged_s2s = strax.merge_peaks(
            peaklets,
            start_merge_at,
            end_merge_at,
            merged=merged,
            max_buffer=max_buffer,
        )
        merged_s2s["type"] = 2

        # if the number of type 20 peaklets inside a peak is larger than the threshold
        # or the area of type 20 peaklets inside a peak is larger than the threshold
        # mark the peaklets as type WIDE_XYPOS_S2_TYPE
        if max_unmerged is not None:
            n_de = []
            area = []
            area_de = []
            for i in range(len(merged_s2s)):
                sl = slice(start_merge_at[i], end_merge_at[i])
                n_de.append(np.sum(~merged[sl]))
                area.append(peaklets["area"][sl][merged[sl]].sum())
                area_de.append(peaklets["area"][sl][~merged[sl]].sum())
            n_de = np.array(n_de)
            area = np.array(area)
            area_de = np.array(area_de)
            # if the number of type 20 peaklets inside a peak is larger than the threshold
            mask = n_de > max_unmerged[0]
            # if the area of type 20 peaklets inside a peak is larger than the (both) threshold(s)
            mask |= (area_de > max_unmerged[1]) & (area_de > area * max_unmerged[2])
            merged_s2s["type"] = np.where(mask, WIDE_XYPOS_S2_TYPE, merged_s2s["type"])

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
    # better constraint the mu in boundaries of waveform
    # because the number of bins is not high enough comprising for computation efficiency
    mu = rough_mu(y, x, rough_mu_bins)

    t0 = y["time"]
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
        y["area"],
        y["area_decile_from_midpoint"],
        y_integrated,
        y["time"],
        y["median_time"],
        x["time"],
        x["median_time"],
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
def rough_mu(y, x, rough_mu_bins):
    """Get rough bins for mu as the integration space."""
    mu = np.linspace(
        min(y["time"], x["time"]) - y["time"],
        max(y["endtime"], x["endtime"]) - y["time"],
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
    y_area,
    y_area_decile_from_midpoint,
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
    y_t = y_area_decile_from_midpoint + y_median_time
    y_t = (y_t[:-1] + y_t[1:]) / 2
    y_nse = y_area / rough_seg
    sigma2 = 2 * sigma**2
    log_sigma = np.log(sigma)
    # dt is median between median time of two peaklets
    dt = np.abs((y_time - x_time) + (y_median_time - x_median_time))
    # because the number of electrons are uniformly distributed in y_t
    log_pdf = -(((y_t - mu[:, None]) ** 2).sum(axis=1) * (y_nse / 10))[:, None] / sigma2
    # complete normal distribution PDF log_sigma * y_nse
    # add log prior dt**2 / 4 / sigma2 + 2 * log_sigma
    # account for the bounded sampling of y np.log(y_integrated) * y_nse
    log_pdf -= np.log(y_integrated) * y_nse + log_sigma * (y_nse + 2) + dt**2 / 4 / sigma2
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
    # do not merge any S2 looks weird
    if not np.all(mask):
        return np.nan
    x_avg = np.average(x[mask], weights=weights[mask])
    y_avg = np.average(y[mask], weights=weights[mask])
    dr = np.sqrt((x - x_avg) ** 2 + (y - y_avg) ** 2)
    dr_avg = np.average(dr[mask], weights=weights[mask])
    return dr_avg
