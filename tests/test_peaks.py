import unittest
import strax
import straxen
import numpy as np
from strax.testutils import run_id
from hypothesis import strategies, given

TEST_DATA_LENGTH =2


class TestComputePeakBasics(unittest.TestCase):
    def setUp(self, context=straxen.contexts.demo):
        self.st = context()
        self.n_top = self.st.config.get('n_top_pmts', 2)
        self.peaks_basics_compute = self.st.get_single_plugin(run_id, 'peak_basics').compute

    @given(strategies.integers(min_value=0,
                               max_value=TEST_DATA_LENGTH-1),
           )
    def test_aft_equals1(self, test_peak_idx):
        """Fill top array with area 1"""
        test_data = self.get_test_data(self.n_top)
        test_data[test_peak_idx]['area_per_channel'][:self.n_top] = 1
        test_data[test_peak_idx]['area'] = np.sum(test_data[test_peak_idx]['area_per_channel'])
        peaks = self.peaks_basics_compute(test_data)
        assert peaks[test_peak_idx]['area_fraction_top'] == 1

    @given(strategies.floats(min_value=0,
                             max_value=2,
                             ),
           strategies.integers(min_value=0,
                               max_value=TEST_DATA_LENGTH-1),
           )
    def test_bad_peak(self, off_by_factor, test_peak_idx):
        if np.isclose(off_by_factor, 1) or np.isclose(off_by_factor, 0):
            return
        test_data = self.get_test_data(self.n_top)
        test_data[test_peak_idx]['area_per_channel'][:self.n_top] = 1
        area = np.sum(test_data[test_peak_idx]['area_per_channel'])
        area *= off_by_factor
        test_data[test_peak_idx]['area'] = area
        self.assertRaises(ValueError, self.peaks_basics_compute, test_data)

    @staticmethod
    def get_test_data(n_top, length=2, sum_wf_samples=10):
        test_data = np.zeros(TEST_DATA_LENGTH,
                             dtype=strax.dtypes.peak_dtype(
                                 n_channels=n_top+1,
                                 n_sum_wv_samples=sum_wf_samples)
                                  )
        test_data['time'] = range(TEST_DATA_LENGTH)
        test_data['time'] *= length * 2
        test_data['dt'] = 1
        test_data['length'] = length
        return test_data
