import numpy as np
from numba import njit

import strax
import straxen


class PeakSEDensity(strax.OverlapWindowPlugin):
    """This plugin is designed to calculate the single electron rate density for each peak. The SE
    rate density is defined as the number of SEs in the vicinity of the peak, divided by the time
    and area of the vicinity.

    References:
        * v0.1.0: xenon:xenonnt:analysis:hot_spot_cut_update

    """

    __version__ = "0.1.0"
    depends_on = ("peak_basics", "peak_positions")
    provides = "peak_se_density"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields + [
        (("Neiboring single-electron rate [Hz/cm^2]", "se_density"), np.float32),
    ]

    se_time_search_window_left = straxen.URLConfig(
        default=1e9, type=int, help="SEs time window searching window, left side [ns]"
    )

    se_time_search_window_right = straxen.URLConfig(
        default=0,
        type=int,
        help="Time searching window right side extension, one max drift time [ns]",
    )

    se_distance_range = straxen.URLConfig(
        default=3.0, help="SEs with radical distance less than this are considered [cm]"
    )

    def get_window_size(self):
        # This method is required by the OverlapWindowPlugin class
        return 2 * (self.se_time_search_window_left + self.se_time_search_window_right)

    def select_se(self, peaks):
        """Function which select single electrons from peaks.

        :param peaks: peaks data contains single electrons.
        :return: single electron peaks data

        """
        mask = peaks["type"] == 2
        mask &= (peaks["area"] > 10) & (peaks["area"] < 80)
        mask &= (peaks["range_50p_area"] > 80) & (peaks["range_50p_area"] < 700)
        return peaks[mask]

    @staticmethod
    @njit()
    def get_se_density(peaks, se_peaks, se_indices, distance_range, time_range):
        """Function to count the SEs in vincinity of each peak, return the count/windows (i.e. SE
        rate density) of each peak.

        :param peaks: array of peaks data
        :param se_peaks: array of single electrons
        :param se_indices: SE indices in vicinity of peaks
        :param distance_range: defined distance range of neighboring
        :param time_range: defined time range of neighboring
        :returns: array of SE rate density for the given peaks

        """
        se_density = np.zeros(len(peaks))
        normalize_factor = time_range / straxen.units.s
        normalize_factor *= np.pi * (distance_range**2)
        for index, peak_i in enumerate(peaks):
            indices = se_indices[index]
            se_in_time = se_peaks[indices[0] : indices[1]]
            radius = np.sqrt(
                (se_in_time["x"] - peak_i["x"]) ** 2 + (se_in_time["y"] - peak_i["y"]) ** 2
            )
            mask_in_circle = radius < distance_range
            se_density[index] = np.sum(mask_in_circle) / normalize_factor
        return se_density

    def compute_se_density(self, peaks, _peaks):
        """Function to calculate the SE rate density for each peak."""
        # select single electrons
        se_peaks = self.select_se(peaks)

        # get single electron indices in peak vicinity
        split_peaks = np.zeros(len(_peaks), dtype=strax.time_fields)
        split_peaks["time"] = _peaks["center_time"] - self.se_time_search_window_left
        split_peaks["endtime"] = _peaks["center_time"] + self.se_time_search_window_right
        split_result = strax.touching_windows(se_peaks, split_peaks)

        # calculate SE density
        _se_density = self.get_se_density(
            _peaks,
            se_peaks,
            split_result,
            self.se_distance_range,
            self.se_time_search_window_right + self.se_time_search_window_left,
        )
        return _se_density

    def compute(self, peaks):
        # sort peaks by center_time
        argsort = np.argsort(peaks["center_time"], kind="mergesort")
        _peaks = np.sort(peaks, order="center_time")

        # prepare output
        se_density = np.zeros(len(peaks))

        # calculate SE density
        _se_density = self.compute_se_density(peaks, _peaks)

        # sort back to original order
        se_density[argsort] = _se_density
        return dict(time=peaks["time"], endtime=strax.endtime(peaks), se_density=se_density)
