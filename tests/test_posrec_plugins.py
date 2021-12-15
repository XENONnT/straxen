import straxen
import unittest
import numpy as np


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class TestPosRecAlgos(unittest.TestCase):
    def setUpClass(cls) -> None:
        cls.target = 'peak_positions_mlp'
        cls.config_name = 'tf_model_mlp'
        cls.field = 'x_mlp'
        cls.run_id = straxen.nt_test_run_id
        cls.st = straxen.nt_test_context()
        cls.st.make(cls.run_id, cls.target)

    def test_set_path(self):
        """Test that we can reconstruct even if we set a hardcoded path"""
        # Manually do a similar thing as the URL config does behind the
        # scenes
        plugin = self.st.get_single_plugin(self.run_id, self.target)
        cmt_config = plugin.config[self.config_name]
        model = straxen.get_correction_from_cmt(self.run_id, cmt_config)
        st_with_hardcoded_path = self.st.new_context()
        file_name = straxen.position_reconstruction.download(model)
        st_with_hardcoded_path.set_config({self.config_name: file_name})
        default_result = self.st.get_array(self.run_id, self.target)
        alt_result = st_with_hardcoded_path.get_array(self.run_id, self.target)
        self.assertTrue(np.alltrue(default_result[self.field], alt_result[self.field]))

    def test_set_to_none(self):
        """Test that we can set the config to None, giving only nan results"""
        st_with_none = self.st.new_context()
        st_with_none.set_config({self.config_name: None})
        alt_result = st_with_none.get_array(self.run_id, self.target)
        self.assertTrue(np.isnan(alt_result[self.field]))
