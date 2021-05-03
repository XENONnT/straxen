import straxen

import numpy as np
import unittest


class TestCreateVetoIntervals(unittest.TestCase):
    def setUp(self):
        dtype = straxen.plugins.veto_events.veto_event_dtype('nveto_eventbumber')
        dtype += straxen.plugins.veto_events.veto_event_positions_dtype()[2:]
        self.dtype = dtype

    def test_empty_inputs(self):
        events = np.zeros(0, self.dtype)
        vetos = straxen.plugins.veto_vetos.create_veto_intervals(events, 0, 0, 0, 10, 10)
        assert len(vetos) == 0, 'Empty input must return empty output!'

    def test_concatenate_overlapping_intervals(self):
        events = np.zeros(4, self.dtype)
        events['area'] = 1

        # First veto interval:
        events[0]['time'] = 2
        events[0]['endtime'] = 3
        events[1]['time'] = 5
        events[1]['endtime'] = 7
        events[2]['time'] = 7
        events[2]['endtime'] = 8

        # Second veto interval:
        events[3]['time'] = 20
        events[3]['endtime'] = 22

        vetos = straxen.plugins.veto_vetos.create_veto_intervals(events,
                                                                 min_area=0,
                                                                 min_hits=0,
                                                                 min_contributing_channels=0,
                                                                 left_extension=1,
                                                                 right_extension=4)
        assert len(vetos) == 2, 'Got the wrong number of veto intervals!'
        assert vetos[0]['time'] == 1, 'First veto event has the wrong time!'
        assert vetos[0]['endtime'] == 12, 'First veto event has the wrong endtime!'
        assert vetos[1]['time'] == 19, 'First veto event has the wrong time!'
        assert vetos[1]['endtime'] == 26, 'First veto event has the wrong endtime!'

    def test_thresholds(self):
        events = np.zeros(1, dtype=self.dtype)
        events['time'] = 2
        events['endtime'] = 3
        events['area'] = 1
        events['n_hits'] = 1
        events['n_contributing_pmt'] = 1

        self._test_threshold_type(events, 'area', 'min_area', 2)
        self._test_threshold_type(events, 'n_hits', 'min_hits', 2)
        self._test_threshold_type(events, 'n_contributing_pmt', 'min_contributing_channels', 2)

    @staticmethod
    def _test_threshold_type(events, field, threshold_type, threshold):
        thresholds = {'min_area': 1,
                      'min_hits': 1,
                      'min_contributing_channels': 1}
        thresholds = {key: (threshold if threshold_type == key else 1) for key in thresholds.keys()}

        vetos = straxen.plugins.veto_vetos.create_veto_intervals(events,
                                                                 **thresholds,
                                                                 left_extension=0,
                                                                 right_extension=0)
        print(events[field], thresholds, vetos)
        assert len(vetos) == 0, f'Vetos for {threshold_type} threshold should be empty since it is below threshold!'

        events[field] = threshold
        vetos = straxen.plugins.veto_vetos.create_veto_intervals(events,
                                                                 **thresholds,
                                                                 left_extension=0,
                                                                 right_extension=0)
        assert len(vetos) == 1, f'{threshold_type} threshold did not work, have a wrong number of vetos!'
        events[field] = 1
