from straxen import rotate_perp_wires, tpc_r, aux_repo, get_resource
import numpy as np
from unittest import TestCase


class TestRotateWires(TestCase):
    """Test that the rotate wires function works or raises usefull errors"""

    def test_rotate_wires(self):
        """Use xy and see that we don't break"""
        x_obs = np.linspace(-tpc_r, -tpc_r, 10)
        y_obs = np.linspace(-tpc_r, -tpc_r, 10)
        rotate_perp_wires(x_obs, y_obs)
        with self.assertRaises(ValueError):
            rotate_perp_wires(x_obs, y_obs[::2])


class TestGetResourceFmt(TestCase):
    """
    Replicate bug with ignored formatting
    github.com/XENONnT/straxen/issues/741
    """
    json_file = aux_repo + '/01809798105f0a6c9efbdfcb5755af087824c234/sim_files/placeholder_map.json'  # noqa

    def test_format(self):
        """
        We did not do this correctly before, so let's make sure to do it right this time
        """
        json_as_text = get_resource(self.json_file, fmt='text')
        self.assertIsInstance(json_as_text, str)
        json_as_dict = get_resource(self.json_file, fmt='json')
        self.assertIsInstance(json_as_dict, dict)
