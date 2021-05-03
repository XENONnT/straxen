import strax
import straxen

import numpy as np


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


class TestCoincidence:

    def test_empty_inputs(self):
        raw_records = np.zeros(0, dtype=strax.time_fields)
        intervals = straxen.coincidence(raw_records)
        assert len(intervals) == 0, 'Empty input should return empty intervals!'

    def test_coincidence(self):
        raw_records = np.zeros(0, dtype=strax.time_fields)
