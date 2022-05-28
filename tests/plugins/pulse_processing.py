"""Run with python tests/plugins/pulse_processing.py"""
from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main


@PluginTestAccumulator.register('test_alt_hitfinder_option')
def test_alternative_hitfinder_options(self: PluginTestCase):
    """Test some old ways of setting the hitfinder options"""
    st = self.st.new_context()
    st.set_config(dict(
        hit_min_amplitude='pmt_commissioning_initial',
        hev_gain_model=("to_pe_placeholder", True),
        tail_veto_threshold=1,
        pmt_pulse_filter=(
            0.012, -0.119, 2.435, -1.271, 0.357, -0.174, -0., -0.036, -0.028, -0.019, -0.025,
            -0.013,
            -0.03, -0.039, -0.005, -0.019, -0.012, -0.015, -0.029, 0.024, -0.007, 0.007, -0.001,
            0.005,
            -0.002, 0.004, -0.002), )
    )

    for target in 'afterpulses records'.split():
        st.make(self.run_id, target)

    # Check some minianalysis with this config
    st.plot_pulses_tpc(self.run_id, seconds_range=(0, 1), plot_median=True)


if __name__ == '__main__':
    run_pytest_from_main()
