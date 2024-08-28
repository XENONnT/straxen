"""Run with python tests/plugins/nv_processing.py."""

from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main
import numpy as np


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


@PluginTestAccumulator.register("test_nveto_recorder_no_trigger_monitoring")
def test_nveto_recorder_no_trigger_monitoring(self: PluginTestCase):
    st = self.st.new_context()
    st.set_config(dict(keep_n_chunks_for_monitoring=1))
    st.make(self.run_id, "records_nv")
    meta_raw = st.get_meta(self.run_id, "raw_records_nv")
    chunks = meta_raw["chunks"]
    start, end = chunks[0][["start", "end"]]
    rr = st.get_array(self.run_id, "raw_records_nv", time_range=(start, end))
    rrc = st.get_array(self.run_id, "raw_records_coin_nv", time_range=(start, end))
    assert np.all(
        rr == rrc
    ), "First chunk of raw data are not the same before and after the software trigger!"


if __name__ == "__main__":
    run_pytest_from_main()
