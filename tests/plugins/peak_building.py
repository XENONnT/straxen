from _core import PluginTestAccumulator, PluginTestCase
import numpy as np


@PluginTestAccumulator.register('test_area_fraction_top')
def test_area_fraction_top(self: PluginTestCase):
    merged_s2s = self.st.get_array(self.run_id, 'merged_s2s', progress_bar=False)
    _area_close_to_area_per_channel = np.isclose(merged_s2s['area'] / np.sum(merged_s2s['area_per_channel'], axis=1), 1)
    assert np.all(_area_close_to_area_per_channel)
