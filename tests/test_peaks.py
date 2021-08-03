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
    """Tests for peak basics plugin"""

    def setUp(self, context=straxen.contexts.demo):
        self.st = context()
        self.n_top = self.st.config.get('n_top_pmts', 2)

        # Make sure that the check is on. Otherwise we cannot test it.
        self.st.set_config({'check_peak_sum_area_rtol': R_TOL_DEFAULT})
        self.peaks_basics_compute = self.st.get_single_plugin(run_id, 'peak_basics').compute

    @settings(deadline=None)
    @given(strategies.integers(min_value=0,
                               max_value=TEST_DATA_LENGTH - 1),
           )
    def test_aft_equals1(self, test_peak_idx):
        """Fill top array with area 1"""
        test_data = self.get_test_peaks(self.n_top)
        test_data[test_peak_idx]['area_per_channel'][:self.n_top] = 1
        test_data[test_peak_idx]['area'] = np.sum(test_data[test_peak_idx]['area_per_channel'])
        peaks = self.peaks_basics_compute(test_data)
        assert peaks[test_peak_idx]['area_fraction_top'] == 1

    @settings(deadline=None)
    @given(strategies.floats(min_value=0,
                             max_value=2,
                             ).filter(_not_close_to_0_or_1),
           strategies.integers(min_value=0,
                               max_value=TEST_DATA_LENGTH - 1,
                               ),
           )
    def test_bad_peak(self, off_by_factor, test_peak_idx):
        """
        Lets deliberately make some data that is not self-consistent to
            run into the error in the test.
        """
        test_data = self.get_test_peaks(self.n_top)
        test_data[test_peak_idx]['area_per_channel'][:self.n_top] = 1
        area = np.sum(test_data[test_peak_idx]['area_per_channel'])

        # Set the area field to a wrong value
        area *= off_by_factor
        test_data[test_peak_idx]['area'] = area
        self.assertRaises(ValueError, self.peaks_basics_compute, test_data)

    @staticmethod
    def get_test_peaks(n_top, length=2, sum_wf_samples=10):
        """Generate some dummy peaks"""
        test_data = np.zeros(TEST_DATA_LENGTH,
                             dtype=strax.dtypes.peak_dtype(
                                 n_channels=n_top + 1,
                                 n_sum_wv_samples=sum_wf_samples)
                             )
        test_data['time'] = range(TEST_DATA_LENGTH)
        test_data['time'] *= length * 2
        test_data['dt'] = 1
        test_data['length'] = length
        return test_data


def create_unique_intervals(size, time_range=(0, 40), allow_zero_length=True):
    """
    Hypothesis stragtegy which creates unqiue time intervals.

    :param size: Number of intervals desired. Can be less if non-unique
        intervals are found.
    :param time_range: Time range in which intervals should be.
    :param allow_zero_length: If true allow zero length intervals.
    """
    strat = strategies.lists(elements=strategies.integers(*time_range),
                             min_size=size*2,
                             max_size=size*2
                             ).map(lambda x: _convert_to_interval(x, allow_zero_length))
    return strat


def _convert_to_interval(time_stamps, allow_zero_length):
    time_stamps = np.sort(time_stamps)
    intervals = np.zeros(len(time_stamps)//2, strax.time_dt_fields)
    intervals['dt'] = 1
    intervals['time'] = time_stamps[::2]
    intervals['length'] = time_stamps[1::2] - time_stamps[::2]

    if not allow_zero_length:
        intervals = intervals[intervals['length'] > 0]
    return np.unique(intervals)


@settings(deadline=None)
@given(create_unique_intervals(10, time_range=(0, 30), allow_zero_length=False),
       create_unique_intervals(5, time_range=(5, 25), allow_zero_length=False)
       )
def test_tag_peaks(peaks, veto_intervals):
    peaks_in_vetos = strax.touching_windows(peaks, veto_intervals)

    tags = np.zeros(len(peaks))
    straxen.plugins.peak_processing.tag_peaks(tags, peaks_in_vetos, 1)

    # Make an additional dummy array to test if function worked:
    dtype = []
    dtype += strax.time_dt_fields
    dtype += [(('peak tag', 'tag'), np.int8)]
    tagged_peaks = np.zeros(len(peaks), dtype)
    tagged_peaks['time'] = peaks['time']
    tagged_peaks['length'] = peaks['length']
    tagged_peaks['dt'] = 1
    tagged_peaks['tag'] = tags

    split_tagged_peaks = strax.split_touching_windows(tagged_peaks, veto_intervals)

    for split_peaks in split_tagged_peaks:
        if not len(split_peaks):
            continue
        assert np.all(split_peaks['tag'] == 1), f'Not all peaks were tagged properly {split_peaks}'


@settings(deadline=None)
@given(create_unique_intervals(10, time_range=(50, 80), allow_zero_length=False),
       create_unique_intervals(5, time_range=(55, 75), allow_zero_length=False)
       )
def test_get_time_to_clostest_veto(peaks, veto_intervals):
    time_differences = straxen.plugins.peak_processing.get_time_to_closest_veto(peaks,
                                                                                veto_intervals
                                                                                )

    if not len(peaks):
        assert not len(time_differences), 'Input is empty but output is not?'

    for ind, p in enumerate(peaks):
        if len(veto_intervals):
            dt = np.concatenate([veto_intervals['time'] - p['time'],
                                 strax.endtime(veto_intervals) - p['time']])
            # If the distance to the curren/next veto interval is identical
            # the function favors positive values. Hence sort and reverse
            # order for the test.
            dt = np.sort(dt)[::-1]
            ind_argmin = np.argmin(np.abs(dt))
            dt = dt[ind_argmin]
        else:
            # If there are not any veto intervalls function will compute clostest
            # difference to +/- 64 bit infinity.
            dt = np.abs(straxen.INFINITY_64BIT_SIGNED - p['time'])

        mes = f'Wrong time difference to closest event. expected: "{dt}" got: "{time_differences[ind]}"'
        assert dt == time_differences[ind], mes
