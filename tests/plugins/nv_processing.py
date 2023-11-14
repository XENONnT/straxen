"""Run with python tests/plugins/nv_processing.py."""

from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main


@PluginTestAccumulator.register("test_nveto_recorder_alt_config")
def test_nveto_recorder_alt_configs(self: PluginTestCase):
    st = self.st.new_context()
    st.set_config(
        dict(
            hit_min_amplitude_nv=[1] * 2120,
            baseline_samples_nb=10,
            coincidence_level_recorder_nv=1,
        )
    )
    st.make(self.run_id, "records_nv")


if __name__ == "__main__":
    run_pytest_from_main()
