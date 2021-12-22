from straxen import rotate_perp_wires, tpc_r
import numpy as np
from unittest import TestCase


class TestCommon(TestCase):
    def test_rotate_wires(self):
        x_obs = np.linspace(-tpc_r, -tpc_r, 10)
        y_obs = np.linspace(-tpc_r, -tpc_r, 10)
        rotate_perp_wires(x_obs, y_obs)
        with self.assertRaises(ValueError):
            rotate_perp_wires(x_obs, y_obs[::2])
