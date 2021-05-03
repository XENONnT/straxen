import strax
import straxen

import numpy as np
import unittest


class TestMergeIntervals:

    def test_empty_intervals(self):
        intervals = np.zeros(0, dtype=strax.time_fields)
        intervals = straxen.merge_intervals(intervals)
        assert len(intervals) == 0, 'Empty input should return empty intervals!'

    def test_merge_overlapping_intervals(self):
        intervals = np.zeros(4, dtype=strax.time_fields)

        # First interval:
        intervals[0]['time'] = 2
        intervals[0]['endtime'] = 4
        intervals[1]['time'] = 3
        intervals[1]['endtime'] = 7
        intervals[2]['time'] = 7
        intervals[2]['endtime'] = 8

        # Second interval:
        intervals[3]['time'] = 20
        intervals[3]['endtime'] = 22

        intervals = straxen.merge_intervals(intervals)

        assert len(intervals) == 2, 'Got the wrong number of intervals!'
        assert intervals[0]['time'] == 2, 'First interval has the wrong time!'
        assert intervals[0]['endtime'] == 8, 'First interval has the wrong endtime!'
        assert intervals[1]['time'] == 20, 'Second interval has the wrong time!'
        assert intervals[1]['endtime'] == 22, 'Second interval has the wrong endtime!'


class TestCoincidence(unittest.TestCase):

    def setUp(self):
        intervals = np.zeros(4, dtype=strax.time_fields)

        intervals[0]['time'] = 3
        intervals[0]['endtime'] = 5
        intervals[1]['time'] = 6
        intervals[1]['endtime'] = 8
        intervals[2]['time'] = 9
        intervals[2]['endtime'] = 10

        intervals[3]['time'] = 38
        intervals[3]['endtime'] = 42
        self.intervals = intervals

    def test_empty_inputs(self):
        intervals = np.zeros(0, dtype=strax.time_fields)
        intervals = straxen.find_coincidence(intervals)
        assert len(intervals) == 0, 'Empty input should return empty intervals!'

    def test_without_coincidence(self):
        coincidence = straxen.find_coincidence(self.intervals, nfold=1, resolving_time=10, pre_trigger=0)
        assert len(coincidence) == 2, 'Have not found the correct number of coincidences'
        assert np.all(coincidence['time'] == (3, 38)), 'Coincidence does not have the correct time'
        assert np.all(coincidence['endtime'] == (19, 48)), 'Coincidence doe snot have the correct time'

    def test_coincidence(self):
        coincidence = straxen.find_coincidence(self.intervals, nfold=3, resolving_time=10, pre_trigger=0)
        assert len(coincidence) == 1, 'Have not found the correct number of coincidences'
        assert coincidence['time'] == 3, 'Coincidence does not have the correct time'
        assert coincidence['endtime'] == 13, 'Coincidence doe snot have the correct time'

        coincidence = straxen.find_coincidence(self.intervals, nfold=3, resolving_time=10, pre_trigger=2)
        assert len(coincidence) == 1, 'Have not found the correct number of coincidences'
        assert coincidence['time'] == 1, 'Coincidence does not have the correct time'
        assert coincidence['endtime'] == 13, 'Coincidence doe snot have the correct time'
