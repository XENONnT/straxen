import straxen
import numpy as np
import unittest


class TestCreateVetoIntervals(unittest.TestCase):
    def setUp(self):
        dtype = straxen.plugins.veto_events.veto_event_dtype('nveto_eventbumber')
        dtype += straxen.plugins.veto_events.veto_event_positions_dtype()[2:]
        self.dtype = dtype

        self.events = np.zeros(4, self.dtype)
        self.events['area'] = 1
        self.events['n_hits'] = 1
        self.events['n_contributing_pmt'] = 1
        self.events['time'] = [2, 5, 7, 20]
        self.events['endtime'] = [3, 7, 8, 22]

    def test_empty_inputs(self):
        events = np.zeros(0, self.dtype)
        vetos = straxen.plugins.veto_veto_regions.create_veto_intervals(events, 0, 0, 0, 10, 10)
        assert len(vetos) == 0, 'Empty input must return empty output!'

    def test_concatenate_overlapping_intervals(self):
        left_extension = 1
        right_extension = 4
        vetos = straxen.plugins.veto_veto_regions.create_veto_intervals(self.events,
                                                                        min_area=0,
                                                                        min_hits=0,
                                                                        min_contributing_channels=0,
                                                                        left_extension=left_extension,
                                                                        right_extension=right_extension)
        assert len(vetos) == 2, 'Got the wrong number of veto intervals!'

        time_is_correct = vetos[0]['time'] == self.events['time'][0] - left_extension
        assert time_is_correct, 'First veto event has the wrong time!'
        time_is_correct = vetos[0]['endtime'] == self.events['endtime'][2] + right_extension
        assert time_is_correct, 'First veto event has the wrong endtime!'

        time_is_correct = vetos[1]['time'] == self.events['time'][-1] - left_extension
        assert time_is_correct, 'Second veto event has the wrong time!'
        time_is_correct = vetos[1]['endtime'] == self.events['endtime'][-1] + right_extension
        assert time_is_correct, 'Second veto event has the wrong endtime!'

    def test_thresholds(self):
        events = np.copy(self.events[:1])

        self._test_threshold_type(events, 'area', 'min_area', 2)
        self._test_threshold_type(events, 'n_hits', 'min_hits', 2)
        self._test_threshold_type(events, 'n_contributing_pmt', 'min_contributing_channels', 2)

    @staticmethod
    def _test_threshold_type(events, field, threshold_type, threshold):
        thresholds = {'min_area': 1,
                      'min_hits': 1,
                      'min_contributing_channels': 1}
        thresholds = {key: (threshold if threshold_type == key else 1) for key in thresholds.keys()}

        vetos = straxen.plugins.veto_veto_regions.create_veto_intervals(events,
                                                                        **thresholds,
                                                                        left_extension=0,
                                                                        right_extension=0)
        print(events[field], thresholds, vetos)
        assert len(vetos) == 0, f'Vetos for {threshold_type} threshold should be empty since it is below threshold!'  # noqa

        events[field] = threshold
        vetos = straxen.plugins.veto_veto_regions.create_veto_intervals(events,
                                                                        **thresholds,
                                                                        left_extension=0,
                                                                        right_extension=0)
        assert len(vetos) == 1, f'{threshold_type} threshold did not work, have a wrong number of vetos!'  # noqa
        events[field] = 1
