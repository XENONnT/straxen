# Run with python tests/plugins/s1_posrec_plugins.py.py
import os
import strax
import straxen
from _core import PluginTestAccumulator, run_pytest_from_main
import numpy as np


@PluginTestAccumulator.register("test_posrec_set_path")
def test_posrec_set_path(
    self,
    target="event_s1_positions_cnn",
    config_name="tf_event_model_s1_cnn",
    field="event_x_s1_cnn",
):
    """Test that we can reconstruct even if we set a hardcoded path."""
    # Manually do a similar thing as the URL config does behind the
    # scenes

    # Get current config
    plugin = self.st.get_single_plugin(self.run_id, target)
    cmt_config = plugin.config[config_name]
    cmt_config_without_tf = cmt_config.replace("tf://", "")
    # Hack URLConfigs to give back intermediate results (this should be easier..)
    st_fixed_path = self.st.new_context()
    st_fixed_path.set_config({config_name: cmt_config_without_tf})
    plugin_fixed = st_fixed_path.get_single_plugin(self.run_id, target)
    file_name = getattr(plugin_fixed, config_name)
    self.assertTrue(os.path.exists(file_name))

    # Now let's see if we can get the same results with both contexts
    set_to_config = f"tf://{file_name}"
    print(f"Setting option to {set_to_config}")
    st_fixed_path.set_config({config_name: set_to_config})
    # Temporarily set min_s1_area_s1_posrec to 0 to allow more events calculated
    default_result = self.st.get_array(self.run_id, target, config=dict(min_s1_area_s1_posrec=0))[
        field
    ]
    alt_result = st_fixed_path.get_array(self.run_id, target, config=dict(min_s1_area_s1_posrec=0))[
        field
    ]
    self.assertTrue(np.all(np.isclose(default_result, alt_result, equal_nan=True)))


@PluginTestAccumulator.register("test_posrec_set_to_none")
def test_posrec_set_to_none(
    self,
    target="event_s1_positions_cnn",
    config_name="tf_event_model_s1_cnn",
    field="event_x_s1_cnn",
):
    """Test that we can set the config to None, giving only nan results."""
    st_with_none = self.st.new_context()
    st_with_none.set_config({config_name: None})
    alt_result = st_with_none.get_array(self.run_id, target)
    self.assertTrue(np.all(np.isnan(alt_result[field])))


@PluginTestAccumulator.register("test_posrec_bad_configs_raising_errors")
def test_posrec_bad_configs_raising_errors(
    self,
    target="event_s1_positions_cnn",
    config_name="tf_event_model_s1_cnn",
):
    """Test that we get the right errors when we set invalid options."""
    dummy_st = self.st.new_context()
    dummy_st.set_config({config_name: "some_path_without_tf_protocol"})

    plugin = dummy_st.get_single_plugin(self.run_id, target)
    with self.assertRaises(ValueError):
        plugin.get_tf_model()

    dummy_st.set_config({config_name: "tf://some_path_that_does_not_exists"})

    plugin = dummy_st.get_single_plugin(self.run_id, target)
    with self.assertRaises(FileNotFoundError):
        plugin.get_tf_model()

    dummy_st.register(straxen.plugins._peaklet_positions_base.PeakletPositionsBase)
    plugin_name = strax.camel_to_snake("PeakletPositionsBase")
    with self.assertRaises(NotImplementedError):
        dummy_st.get_single_plugin(self.run_id, plugin_name)


if __name__ == "__main__":
    # Run with python tests/plugins/s1_posrec_plugins.py
    run_pytest_from_main()
