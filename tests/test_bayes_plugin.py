import strax
import straxen
import unittest
import numpy as np


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class TestBayesPlugin(unittest.TestCase):

    def setUp(self):

        st = straxen.test_utils.nt_test_context()
        st.register(straxen.BayesPeakletClassification)
        self.target = 'peaklet_classification_bayes'
        self.run_id = straxen.test_utils.nt_test_run_id
        self.st = st

    def test_bins(self):
        """
        Test bins are monotonic, if no, issue w/ config file (training problem)
        """
        plugin = self.st.get_single_plugin(self.run_id, self.target)
        bins = plugin.bayes_config_file['bins']
        waveform_bin_edges = bins[0, :][bins[0, :] > -1]
        waveform_num_bin_edges = len(waveform_bin_edges)
        monotonic = np.all(np.diff(waveform_bin_edges) > 0)
        self.assertTrue(monotonic)

    def test_inference(self):
        """
        Test inference, ln prob <= 0
        """
        self.st.make(self.run_id, self.target)
        bayes = self.st.get_array(self.run_id, self.target)
        prob = np.where(bayes['s1_ln_prob'] > 1.0)
        assert len(prob) == 0, "ln prob gratter than one, impossible"
