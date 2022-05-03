from _core import PluginTestAccumulator, PluginTestCase
import numpy as np


@PluginTestAccumulator.register('test_alt_hitfinder_option')
def test_alternative_hitfinder_options(self: PluginTestCase):
    """Test some old ways of setting the hitfinder options"""
    st = self.st
    st.set_config(dict(hit_min_amplitude='pmt_commissioning_initial'))
    # Check some minianalysis with this config
    st.plot_pulses_top(self.run_id, seconds_range=(0,1), plot_median=True)

    for target in 'afterpulses records':
        st.make(self.run_id, target)
