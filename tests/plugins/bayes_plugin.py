"""Run with python tests/plugins/bayes_plugin.py."""
from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main
import numpy as np


@PluginTestAccumulator.register("test_bayes_bins")
def test_bayes_bins(self: PluginTestCase):
    """Test bins are monotonic, if no, issue w/ config file (training problem)"""
    target = "peak_classification_bayes"
    plugin = self.st.get_single_plugin(self.run_id, target)
    bins = plugin.bayes_config_file["bins"]
    waveform_bin_edges = bins[0, :][bins[0, :] > -1]
    monotonic = np.all(np.diff(waveform_bin_edges) > 0)
    self.assertTrue(monotonic)


@PluginTestAccumulator.register("test_bayes_inference")
def test_bayes_inference(self: PluginTestCase):
    """Test inference, ln prob <= 0."""
    target = "peak_classification_bayes"
    self.st.make(self.run_id, "peaks")
    self.st.make(self.run_id, target)
    bayes = self.st.get_array(self.run_id, target)
    assert not np.any(bayes["ln_prob_s1"] > 0)


if __name__ == "__main__":
    run_pytest_from_main()
