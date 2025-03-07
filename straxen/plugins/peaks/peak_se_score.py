import numpy as np
from numba import njit

import strax
import straxen


class PeakSEScore(strax.OverlapWindowPlugin):
    """This plugin is designed to calculate a score based on the position relations to single
    electrons.

    References:
        * v0.2.0: xenon:xenonnt:analysis:hot_spot_cut_summary
        * v0.4.0: xenon:xenonnt:ac:sr1:hotspot_veto_cut:wimp_roi
        * position resolution: xenon:xenonnt:svetter:data_driven_resolution_sr1a0_vs_sr1a1

    """

    __version__ = "0.4.0"
    depends_on = ("peak_basics", "peak_positions")
    provides = "peak_se_score"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields + [
        (
            ("Sum of the PDF of the SE position nearby probability.", "se_score"),
            np.float32,
        ),
    ]

    se_time_search_window_left = straxen.URLConfig(
        default=5e9, type=int, help="SEs time window searching window, left side [ns]"
    )

    se_time_search_window_right = straxen.URLConfig(
        default=0,
        type=int,
        help="Time searching window right side extension, one max drift time [ns]",
    )

    se_selection_area_roi = straxen.URLConfig(
        default=[10, 80],
        type=list,
        help="Area range for single electron selection.[PE]",
    )

    se_selection_width_roi = straxen.URLConfig(
        default=[80, 700],
        type=list,
        help="Area range for single electron selection.[PE]",
    )

    para_a_naive = straxen.URLConfig(
        default=1,
        help="Place holder for Parameter A in the position resolution function.",
    )

    para_b_naive = straxen.URLConfig(
        default=1,
        help="Place holder for Parameter B in the position resolution function.",
    )

    n_se_score_group = straxen.URLConfig(
        default=1e7,
        type=(int, float),
        track=False,
        help="Number of SEs to be grouped for less time usage.",
    )

    def get_window_size(self):
        # This method is required by the OverlapWindowPlugin class
        return 2 * (self.se_time_search_window_left + self.se_time_search_window_right)

    def setup(self):
        self._para_a = self.para_a_naive
        self._para_b = self.para_b_naive

    def select_se(self, peaks):
        """Function which select single electrons from peaks.

        :param peaks: peaks data contains single electrons.
        :return: single electron peaks data

        """
        mask = peaks["type"] == 2
        mask &= (peaks["area"] > self.se_selection_area_roi[0]) & (
            peaks["area"] < self.se_selection_area_roi[1]
        )
        mask &= (peaks["range_50p_area"] > self.se_selection_width_roi[0]) & (
            peaks["range_50p_area"] < self.se_selection_width_roi[1]
        )
        mask &= peaks["area_fraction_top"] != 0
        return mask

    @staticmethod
    @njit
    def get_se_count_and_pdf_sum(peaks, se_peaks, se_indices, para_a, para_b):
        """Function to calculate the single electron (SE) score for each peak.

        This function calculates the probability of nearby SE events for each peak
        based on their spatial and temporal proximity. The probability is calculated
        using a Gaussian distribution, where the standard deviation is the combined
        position resolution of the peak and the SE event.

        Parameters:
        :peaks (np.ndarray): Array of peaks.
        :se_peaks (np.ndarray): Array of single electron peaks.
        :se_indices (np.ndarray): Array of ind for SE that are
        in temporal proximity to each peak.
        :para_a (float): Parameter 'a' used in the calculation of position resolution.
        :para_b (float): Parameter 'b' used in the calculation of position resolution.

        :return: np.ndarray: Array of probabilities of nearby SE events for each peak.

        """
        pk_area_top = peaks["area"] * peaks["area_fraction_top"]
        pk_resolution = para_a / np.sqrt(pk_area_top) + para_b
        se_area_top = se_peaks["area"] * se_peaks["area_fraction_top"]
        se_resolution = para_a / np.sqrt(se_area_top) + para_b

        n_se = se_indices[:, 1] - se_indices[:, 0]
        indices = np.repeat(np.arange(len(peaks)), n_se)
        pk_x = np.repeat(peaks["x"], n_se)
        se_x = np.hstack([se_peaks["x"][ind[0] : ind[1]] for ind in se_indices])
        assert len(indices) == len(pk_x) == len(se_x)
        pk_y = np.repeat(peaks["y"], n_se)
        se_y = np.hstack([se_peaks["y"][ind[0] : ind[1]] for ind in se_indices])
        pk_resolution_expanded = np.repeat(pk_resolution, n_se)
        se_resolution_expanded = np.hstack([se_resolution[ind[0] : ind[1]] for ind in se_indices])

        combined_resolution = np.sqrt(se_resolution_expanded**2 + pk_resolution_expanded**2)
        d_squre = (pk_x - se_x) ** 2 + (pk_y - se_y) ** 2
        # 2D Gaussian with zero off-diagonal covairance matirx elements
        constant = 1 / (2 * np.pi * combined_resolution**2)
        exponent = np.exp(-1 / 2 * (d_squre / combined_resolution**2))
        se_nearby_probability = np.bincount(indices, weights=constant * exponent)
        return se_nearby_probability

    def compute_se_score(self, peaks, _peaks):
        """Function to calculate the SE score for each peak."""
        # select single electrons
        mask = self.select_se(peaks)
        se_peaks = peaks[mask]

        # get single electron ind in peak vicinity
        split_peaks = np.zeros(len(_peaks), dtype=strax.time_fields)
        split_peaks["time"] = _peaks["center_time"] - self.se_time_search_window_left
        split_peaks["endtime"] = _peaks["center_time"] + self.se_time_search_window_right
        se_indices = strax.touching_windows(se_peaks, split_peaks)

        # group SE indices
        if self.n_se_score_group <= 0:
            raise ValueError(
                "The size of SE group should be larger than 0, but got {self.n_se_score_group}."
            )
        n_se = se_indices[:, 1] - se_indices[:, 0]
        n_seen = 0
        se_group = [0]
        for i, n in enumerate(n_se):
            n_seen += n
            if n_seen > self.n_se_score_group:
                se_group.append(i + 1)
                n_seen = 0
        if len(n_se) not in se_group:
            se_group.append(len(n_se))

        # get SE score
        se_nearby_probability = np.zeros(len(peaks))
        for i, j in zip(se_group[:-1], se_group[1:]):
            se_nearby_probability[i:j] = self.get_se_count_and_pdf_sum(
                _peaks[i:j],
                se_peaks,
                strax.touching_windows(se_peaks, split_peaks[i:j]),
                self._para_a,
                self._para_b,
            )
        return se_nearby_probability

    def compute(self, peaks):
        # sort peaks by center_time
        argsort = strax.stable_argsort(peaks["center_time"])
        _peaks = strax.stable_sort(peaks, order="center_time")
        mask_nan = np.isnan(_peaks["x"]) | np.isnan(_peaks["y"])
        # prepare output
        se_nearby_probability = np.zeros(len(peaks))
        _se_nearby_probability = np.full(len(peaks), np.nan)
        # calculate SE Score
        _se_nearby_probability[~mask_nan] = self.compute_se_score(peaks, _peaks[~mask_nan])
        # sort back to original order
        se_nearby_probability[argsort] = _se_nearby_probability
        return dict(
            time=peaks["time"],
            endtime=strax.endtime(peaks),
            se_score=se_nearby_probability,
        )
