import strax
import straxen
import numpy as np
import unittest


class TestMergeIntervals(unittest.TestCase):
    def setUp(self):
        self.intervals = np.zeros(4, dtype=strax.time_fields)
        self.intervals["time"] = [2, 3, 7, 20]
        self.intervals["endtime"] = [4, 7, 8, 22]

    def test_empty_intervals(self):
        intervals = np.zeros(0, dtype=strax.time_fields)
        intervals = straxen.merge_intervals(intervals)
        assert len(intervals) == 0, "Empty input should return empty intervals!"

    def test_merge_overlapping_intervals(self):
        intervals = straxen.merge_intervals(self.intervals)

        assert len(intervals) == 2, "Got the wrong number of intervals!"

        time_is_correct = intervals[0]["time"] == self.intervals["time"][0]
        assert time_is_correct, "First interval has the wrong time!"
        time_is_correct = intervals[0]["endtime"] == self.intervals["endtime"][-2]
        assert time_is_correct, "First interval has the wrong endtime!"

        time_is_correct = intervals[-1]["time"] == self.intervals["time"][-1]
        assert time_is_correct, "Second interval has the wrong time!"
        time_is_correct = intervals[-1]["endtime"] == self.intervals["endtime"][-1]
        assert time_is_correct, "Second interval has the wrong endtime!"


class TestCoincidence(unittest.TestCase):
    def setUp(self):
        self.intervals = np.zeros(8, dtype=strax.time_fields)
        self.intervals["time"] = [3, 6, 9, 12, 15, 18, 21, 38]
        self.intervals["endtime"] = [5, 8, 10, 13, 16, 19, 23, 42]

    def test_empty_inputs(self):
        intervals = np.zeros(0, dtype=strax.time_fields)
        intervals = straxen.find_coincidence(intervals)
        assert len(intervals) == 0, "Empty input should return empty intervals!"

    def test_without_coincidence(self):
        resolving_time = 10
        truth_time = np.array([self.intervals["time"][0], self.intervals["time"][-1]])
        truth_endtime = (
            np.array([self.intervals["time"][-2], self.intervals["time"][-1]]) + resolving_time
        )
        self._test_coincidence(
            resolving_time=resolving_time,
            coincidence=1,
            pre_trigger=0,
            n_concidences_truth=2,
            times_truth=truth_time,
            endtime_truth=truth_endtime,
        )
        pre_trigger = 2
        self._test_coincidence(
            resolving_time=resolving_time,
            coincidence=1,
            pre_trigger=pre_trigger,
            n_concidences_truth=2,
            times_truth=truth_time - pre_trigger,
            endtime_truth=truth_endtime,
        )

    def test_even_fold(self):
        resolving_time = 10
        self._test_coincidence(
            resolving_time=resolving_time,
            coincidence=2,
            pre_trigger=0,
            n_concidences_truth=1,
            times_truth=self.intervals["time"][0],
            endtime_truth=self.intervals["time"][-3] + resolving_time,
        )
        pre_trigger = 2
        self._test_coincidence(
            resolving_time=resolving_time,
            coincidence=2,
            pre_trigger=pre_trigger,
            n_concidences_truth=1,
            times_truth=self.intervals["time"][0] - pre_trigger,
            endtime_truth=self.intervals["time"][-3] + resolving_time,
        )

        self._test_coincidence(
            resolving_time=resolving_time,
            coincidence=4,
            pre_trigger=pre_trigger,
            n_concidences_truth=1,
            times_truth=self.intervals["time"][0] - pre_trigger,
            endtime_truth=self.intervals["time"][-5] + resolving_time,
        )

    def test_odd_fold(self):
        resolving_time = 10
        self._test_coincidence(
            resolving_time=resolving_time,
            coincidence=3,
            pre_trigger=0,
            n_concidences_truth=1,
            times_truth=self.intervals["time"][0],
            endtime_truth=self.intervals["time"][-4] + resolving_time,
        )
        pre_trigger = 2
        self._test_coincidence(
            resolving_time=resolving_time,
            coincidence=3,
            pre_trigger=pre_trigger,
            n_concidences_truth=1,
            times_truth=self.intervals["time"][0] - pre_trigger,
            endtime_truth=self.intervals["time"][-4] + resolving_time,
        )

        self._test_coincidence(
            resolving_time=resolving_time,
            coincidence=5,
            pre_trigger=pre_trigger,
            n_concidences_truth=0,
            times_truth=self.intervals["time"][:0],
            endtime_truth=self.intervals["time"][:0],
        )

    def _test_coincidence(
        self,
        resolving_time,
        coincidence,
        pre_trigger,
        n_concidences_truth,
        times_truth,
        endtime_truth,
    ):
        coincidence = straxen.find_coincidence(
            self.intervals,
            nfold=coincidence,
            resolving_time=resolving_time,
            pre_trigger=pre_trigger,
        )
        number_coincidence_correct = len(coincidence) == n_concidences_truth
        assert number_coincidence_correct, "Have not found the correct number of coincidences"

        time_is_correct = np.all(coincidence["time"] == times_truth)
        assert time_is_correct, "Coincidence does not have the correct time"

        endtime_is_correct = np.all(coincidence["endtime"] == endtime_truth)
        print(coincidence["endtime"], endtime_truth)
        assert endtime_is_correct, "Coincidence does not have the correct endtime"


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_nv_for_dummy_rr():
    """Basic test to run the nv rr for dummy raw-records."""
    st = straxen.test_utils.nt_test_context(deregister=())
    st.context_config["forbid_creation_of"] = tuple()
    st.register(straxen.test_utils.DummyRawRecords)
    st.make(straxen.test_utils.nt_test_run_id, "hitlets_nv")
