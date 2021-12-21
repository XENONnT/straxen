import os

import strax
import straxen
import unittest
import numpy as np


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class TestPosRecAlgorithms(unittest.TestCase):
    """
    Test several options for our posrec plugins
    """
    @classmethod
    def setUpClass(cls) -> None:
        cls.target = 'peak_positions_mlp'
        cls.config_name = 'tf_model_mlp'
        cls.field = 'x_mlp'
        cls.run_id = straxen.test_utils.nt_test_run_id
        cls.st = straxen.test_utils.nt_test_context()
        cls.st.make(cls.run_id, cls.target)

    def test_set_path(self):
        """Test that we can reconstruct even if we set a hardcoded path"""
        # Manually do a similar thing as the URL config does behind the
        # scenes

        # Get current config
        plugin = self.st.get_single_plugin(self.run_id, self.target)
        cmt_config = plugin.config[self.config_name]
        cmt_config_without_tf = cmt_config.replace('tf://', '')

        # Hack URLConfigs to give back intermediate results (this should be easier..)
        st_fixed_path = self.st.new_context()
        st_fixed_path.set_config({self.config_name: cmt_config_without_tf})
        plugin_fixed = st_fixed_path.get_single_plugin(self.run_id, self.target)
        file_name = getattr(plugin_fixed, self.config_name)
        self.assertTrue(os.path.exists(file_name))

        # Now let's see if we can get the same results with both contexts
        set_to_config = f'tf://{file_name}'
        print(f'Setting option to {set_to_config}')
        st_fixed_path.set_config({self.config_name: set_to_config})
        default_result = self.st.get_array(self.run_id, self.target)[self.field]
        alt_result = st_fixed_path.get_array(self.run_id, self.target)[self.field]
        self.assertTrue(np.all(np.isclose(default_result, alt_result)))

    def test_set_to_none(self):
        """Test that we can set the config to None, giving only nan results"""
        st_with_none = self.st.new_context()
        st_with_none.set_config({self.config_name: None})
        alt_result = st_with_none.get_array(self.run_id, self.target)
        self.assertTrue(np.all(np.isnan(alt_result[self.field])))

    def test_bad_configs_raising_errors(self):
        """Test that we get the right errors when we set invalid options"""
        dummy_st = self.st.new_context()
        dummy_st.set_config({self.config_name: 'some_path_without_tf_protocol'})

        plugin = dummy_st.get_single_plugin(self.run_id, self.target)
        with self.assertRaises(ValueError):
            plugin.get_tf_model()

        dummy_st.set_config({self.config_name: 'tf://some_path_that_does_not_exists'})

        plugin = dummy_st.get_single_plugin(self.run_id, self.target)
        with self.assertRaises(FileNotFoundError):
            plugin.get_tf_model()

        dummy_st.register(straxen.position_reconstruction.PeakPositionsBaseNT)
        plugin_name = strax.camel_to_snake('PeakPositionsBaseNT')
        with self.assertRaises(NotImplementedError):
            dummy_st.get_single_plugin(self.run_id, plugin_name)
