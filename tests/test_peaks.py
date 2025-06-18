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


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class TestComputePeakBasics(unittest.TestCase):
    """Tests for peak basics plugin."""

    def setUp(self):
        self.st = straxen.test_utils.nt_test_context("xenonnt_online")
        self.n_top = self.st.config.get("n_top_pmts", 2)

        # Make sure that the check is on. Otherwise we cannot test it.
        self.st.set_config(
            {
                "n_top_pmts": self.n_top,
                "check_peak_sum_area_rtol": R_TOL_DEFAULT,
            }
        )
        self.peaks_basics = self.st.get_single_plugin(run_id, "peak_basics")
        self.peaklet_classification = self.st.get_single_plugin(run_id, "peaklet_classification")
        self.dtype = strax.merged_dtype(
            (
                np.dtype(strax.peak_dtype(n_channels=self.n_top + 1, n_sum_wv_samples=10)),
                self.peaklet_classification.dtype_for("peaklet_classification"),
            )
        )

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
        test_data = self.get_test_peaks()
        test_data[test_peak_idx]["area_per_channel"][: self.n_top] = 1
        area = np.sum(test_data[test_peak_idx]["area_per_channel"])

        # Set the area field to a wrong value
        area *= off_by_factor
        test_data[test_peak_idx]["area"] = area
        self.assertRaises(ValueError, self.peaks_basics.compute, test_data)

    def get_test_peaks(self, length=2):
        """Generate some dummy peaks."""
        test_data = np.zeros(
            TEST_DATA_LENGTH,
            dtype=self.dtype,
        )
        test_data["time"] = range(TEST_DATA_LENGTH)
        test_data["time"] *= length * 2
        test_data["dt"] = 1
        test_data["length"] = length
        return test_data
