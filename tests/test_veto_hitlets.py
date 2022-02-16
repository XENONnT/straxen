import strax
import straxen
import numpy as np

import unittest


class TestRemoveSwtichedOffChannels(unittest.TestCase):
    def setUp(self):
        self.channel_range = (10, 19)
        self.to_pe = np.zeros(20)
        self.to_pe[10:17] = 1

    def test_empty_inputs(self):
        hits = np.zeros(0, strax.hit_dtype)
        hits = straxen.veto_hitlets.remove_switched_off_channels(hits,
                                                                 self.to_pe)
        assert not len(hits), 'Empty input should return an empty result.'

    def test_return(self):
        hits = np.zeros(2, strax.hit_dtype)
        hits[0]['channel'] = 15
        hits[1]['channel'] = 18
        hits_returned = straxen.veto_hitlets.remove_switched_off_channels(hits,
                                                                          self.to_pe)
        assert hits_returned['channel'] == 15, 'Returned a wrong channel.'

        self.to_pe[:] = 1
        hits_returned = straxen.veto_hitlets.remove_switched_off_channels(hits,
                                                                          self.to_pe)
        assert len(hits_returned) == 2, 'Did not return all channels.'
