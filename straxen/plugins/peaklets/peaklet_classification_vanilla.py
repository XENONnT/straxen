from typing import Union

import numpy as np
import strax
import straxen


export, __all__ = strax.exporter()


@export
class PeakletClassificationVanilla(strax.Plugin):
    """Classify peaklets as unknown, S1, or S2."""

    __version__ = "3.0.4"

    depends_on = "peaklets"
    provides: Union[str, tuple] = "peaklet_classification"
    dtype = strax.peak_interval_dtype + [("type", np.int8, "Classification of the peak(let)")]

    s1_risetime_area_parameters = straxen.URLConfig(
        default=(50, 80, 12),
        type=(list, tuple),
        help="norm, const, tau in the empirical boundary in the risetime-area plot",
    )

    s1_risetime_aft_parameters = straxen.URLConfig(
        default=(-1, 2.6),
        type=(list, tuple),
        help=(
            "Slope and offset in exponential of emperical boundary in the rise time-AFT "
            "plot. Specified as (slope, offset)"
        ),
    )

    s1_flatten_threshold_aft = straxen.URLConfig(
        default=(0.6, 100),
        type=(tuple, list),
        help=(
            "Threshold for AFT, above which we use a flatted boundary for rise time"
            "Specified values: (AFT boundary, constant rise time)."
        ),
    )

    s1_max_rise_time_post100 = straxen.URLConfig(
        default=200, type=(int, float), help="Maximum S1 rise time for > 100 PE [ns]"
    )

    s1_min_coincidence = straxen.URLConfig(
        default=2, type=int, help="Minimum tight coincidence necessary to make an S1"
    )

    s2_min_pmts = straxen.URLConfig(
        default=4, type=int, help="Minimum number of PMTs contributing to an S2"
    )

    @staticmethod
    def upper_rise_time_area_boundary(area, norm, const, tau):
        """Function which determines the upper boundary for the rise-time for a given area."""
        return norm * np.exp(-area / tau) + const

    @staticmethod
    def upper_rise_time_aft_boundary(aft, slope, offset, aft_boundary, flat_threshold):
        """Function which computes the upper rise time boundary as a function of area fraction
        top."""
        res = 10 ** (slope * aft + offset)
        res[aft >= aft_boundary] = flat_threshold
        return res

    def compute(self, peaklets):
        ptype = np.zeros(len(peaklets), dtype=np.int8)

        # Properties needed for classification:
        rise_time = -peaklets["area_decile_from_midpoint"][:, 1]
        n_channels = (peaklets["area_per_channel"] > 0).sum(axis=1)

        is_large_s1 = peaklets["area"] >= 100
        is_large_s1 &= rise_time <= self.s1_max_rise_time_post100
        is_large_s1 &= peaklets["tight_coincidence"] >= self.s1_min_coincidence

        is_small_s1 = peaklets["area"] < 100
        is_small_s1 &= rise_time < self.upper_rise_time_area_boundary(
            peaklets["area"],
            *self.s1_risetime_area_parameters,
        )

        is_small_s1 &= rise_time < self.upper_rise_time_aft_boundary(
            peaklets["area_fraction_top"],
            *self.s1_risetime_aft_parameters,
            *self.s1_flatten_threshold_aft,
        )

        is_small_s1 &= peaklets["tight_coincidence"] >= self.s1_min_coincidence

        ptype[is_large_s1 | is_small_s1] = 1

        is_s2 = n_channels >= self.s2_min_pmts
        is_s2[is_large_s1 | is_small_s1] = False
        ptype[is_s2] = 2

        peaklets_classification = np.zeros(len(peaklets), dtype=self.dtype)
        peaklets_classification["type"] = ptype
        peaklets_classification["time"] = peaklets["time"]
        peaklets_classification["dt"] = peaklets["dt"]
        peaklets_classification["length"] = peaklets["length"]
        peaklets_classification["channel"] = -1

        return peaklets_classification
