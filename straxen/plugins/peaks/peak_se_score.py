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
        type=int,
        help="Area range for single electron selection.[PE]",
    )

    se_selection_width_roi = straxen.URLConfig(
        default=[80, 700],
        type=int,
        help="Area range for single electron selection.[PE]",
    )

    para_a = straxen.URLConfig(
        default=1,
        help="Place holder for Parameter A in the position resolution function.",
    )

    para_b = straxen.URLConfig(
        default=1,
        help="Place holder for Parameter B in the position resolution function.",
    )

    def get_window_size(self):
        # This method is required by the OverlapWindowPlugin class
        return 2 * (self.se_time_search_window_left + self.se_time_search_window_right)

    def setup(self):
        self._para_a = self.para_a
        self._para_b = self.para_b

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
        return mask

    @staticmethod
    @njit
    def get_se_count_and_pdf_sum(peaks, se_peaks, se_indices, para_a, para_b, eps):
        se_nearby_probability = np.zeros(len(peaks))
        for index, peak_i in enumerate(peaks):
            indices = se_indices[index]
            se_in_time = se_peaks[indices[0] : indices[1]]
            peak_area_top = peak_i["area"] * (peak_i["area_fraction_top"] + eps)
            se_area_top = se_in_time["area"] * (se_in_time["area_fraction_top"] + eps)
            peak_position_resolution = para_a * 1 / np.sqrt(peak_area_top) + para_b
            se_position_resolution = para_a * 1 / np.sqrt(se_area_top) + para_b
            combined_position_resolution = np.sqrt(
                se_position_resolution**2 + peak_position_resolution**2
            )
            d_squre = (se_in_time["x"] - peak_i["x"]) ** 2 + (se_in_time["y"] - peak_i["y"]) ** 2
            constant = 1 / (2 * np.pi * combined_position_resolution**2)
            exponent = np.exp(-1 / 2 * (d_squre / combined_position_resolution**2))
            _se_nearby_prob = constant * exponent
            se_nearby_probability[index] = np.sum(_se_nearby_prob)
        return se_nearby_probability

    def compute_se_score(self, peaks, _peaks):
        """Function to calculate the SE score for each peak."""
        # select single electrons
        mask = self.select_se(peaks)
        se_peaks = peaks[mask]
        # get single electron indices in peak vicinity
        split_peaks = np.zeros(len(_peaks), dtype=strax.time_fields)
        split_peaks["time"] = _peaks["center_time"] - self.se_time_search_window_left
        split_peaks["endtime"] = _peaks["center_time"] + self.se_time_search_window_right
        split_result = strax.touching_windows(se_peaks, split_peaks)
        # get se score
        eps = np.finfo(float).eps
        _se_nearby_probability = self.get_se_count_and_pdf_sum(
            _peaks,
            se_peaks,
            split_result,
            self._para_a[self._sr_phase],
            self._para_b[self._sr_phase],
            eps,
        )
        return _se_nearby_probability

    def compute(self, peaks):
        # sort peaks by center_time
        argsort = np.argsort(peaks["center_time"], kind="mergesort")
        _peaks = np.sort(peaks, order="center_time")
        # prepare output
        se_nearby_probability = np.zeros(len(peaks))
        # calculate SE Score
        _se_nearby_probability = self.compute_se_score(peaks, _peaks)
        # sort back to original order
        se_nearby_probability[argsort] = _se_nearby_probability

        return dict(
            time=peaks["time"],
            endtime=strax.endtime(peaks),
            se_score=se_nearby_probability,
        )
