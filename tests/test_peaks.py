import strax
import straxen
import numpy as np
import unittest
from strax.testutils import run_id
from hypothesis import strategies, given, settings

TEST_DATA_LENGTH = 3
R_TOL_DEFAULT = 1e-5


def _not_close_to_0_or_1(x, rtol=R_TOL_DEFAULT):
    return not (np.isclose(x, 1, rtol=rtol) or np.isclose(x, 0, rtol=rtol))


class TestComputePeakBasics(unittest.TestCase):
    """Tests for peak basics plugin."""

    def setUp(self, context=straxen.contexts.demo):
        self.st = context()
        self.n_top = self.st.config.get("n_top_pmts", 2)

        # Make sure that the check is on. Otherwise we cannot test it.
        self.st.set_config({"check_peak_sum_area_rtol": R_TOL_DEFAULT})
        self.peaks_basics_compute = self.st.get_single_plugin(run_id, "peak_basics").compute

    @settings(deadline=None)
    @given(
        strategies.integers(min_value=0, max_value=TEST_DATA_LENGTH - 1),
    )
    def test_aft_equals1(self, test_peak_idx):
        """Fill top array with area 1."""
        test_data = self.get_test_peaks(self.n_top)
        test_data[test_peak_idx]["area_per_channel"][: self.n_top] = 1
        test_data[test_peak_idx]["area"] = np.sum(test_data[test_peak_idx]["area_per_channel"])
        peaks = self.peaks_basics_compute(test_data)
        assert peaks[test_peak_idx]["area_fraction_top"] == 1

    @settings(deadline=None)
    @given(
        strategies.floats(
            min_value=0,
            max_value=2,
        ).filter(_not_close_to_0_or_1),
        strategies.integers(
            min_value=0,
            max_value=TEST_DATA_LENGTH - 1,
        ),
    )
    def test_bad_peak(self, off_by_factor, test_peak_idx):
        """Lets deliberately make some data that is not self-consistent to run into the error in the
        test."""
        test_data = self.get_test_peaks(self.n_top)
        test_data[test_peak_idx]["area_per_channel"][: self.n_top] = 1
        area = np.sum(test_data[test_peak_idx]["area_per_channel"])

        # Set the area field to a wrong value
        area *= off_by_factor
        test_data[test_peak_idx]["area"] = area
        self.assertRaises(ValueError, self.peaks_basics_compute, test_data)

    @staticmethod
    def get_test_peaks(n_top, length=2, sum_wf_samples=10):
        """Generate some dummy peaks."""
        test_data = np.zeros(
            TEST_DATA_LENGTH,
            dtype=strax.dtypes.peak_dtype(n_channels=n_top + 1, n_sum_wv_samples=sum_wf_samples),
        )
        test_data["time"] = range(TEST_DATA_LENGTH)
        test_data["time"] *= length * 2
        test_data["dt"] = 1
        test_data["length"] = length
        return test_data
