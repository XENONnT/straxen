from unittest import TestCase, skipIf
import numpy as np
from straxen import utilix_is_configured, get_resource
from straxen import InterpolatingMap, save_interpolation_formatted_map


class TestItpMaps(TestCase):
    def open_map(self, map_name, fmt, method="WeightedNearestNeighbors"):
        map_data = get_resource(map_name, fmt=fmt)
        m = InterpolatingMap(map_data, method=method)
        self.assertTrue(m is not None)

    @skipIf(not utilix_is_configured(), "Cannot download maps without db access")
    def test_lce_map(self):
        self.open_map("XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz", fmt="json.gz")

    @property
    def _map(self):
        """See https://github.com/XENONnT/straxen/pull/757"""
        map = {
            "coordinate_system": [
                [-18.3, -31.7, -111.5],
                [36.6, -0.0, -111.5],
                [-18.3, 31.7, -111.5],
                [-18.3, -31.7, -37.5],
                [36.6, -0.0, -37.5],
                [-18.3, 31.7, -37.5],
            ],
            "description": "Array_valued dummy map with lists",
            "name": "Dummy map",
            "map": [
                [1.7, 11.1],
                [1.7, 11.1],
                [1.7, 11.0],
                [3.3, 5.7],
                [3.3, 5.8],
                [3.3, 5.7],
            ],
        }
        return map

    def test_array_valued(self):
        filename = "test_array_valued.pkl"
        save_interpolation_formatted_map(
            self._map["map"],
            self._map["coordinate_system"],
            filename=filename,
        )
        filename_quantized = "test_array_valued_quantized.pkl"
        save_interpolation_formatted_map(
            self._map["map"],
            self._map["coordinate_system"],
            filename=filename_quantized,
            quantum=0.001,
        )
        filename_quantized_gzip = "test_array_valued_quantized.pkl.gz"
        save_interpolation_formatted_map(
            self._map["map"],
            self._map["coordinate_system"],
            filename=filename_quantized_gzip,
            quantum=0.001,
        )

        # Let's do something easy, check if one fixed point yields the
        # same result if not, our interpolation map depends on the
        # straxen version?! That's bad!
        itp_maps = [
            InterpolatingMap(self._map),
            InterpolatingMap(get_resource(filename, fmt="pkl")),
            InterpolatingMap(get_resource(filename_quantized, fmt="pkl")),
            InterpolatingMap(get_resource(filename_quantized_gzip, fmt="pkl.gz")),
        ]
        for itp_map in itp_maps:
            map_at_random_point = itp_map([[0, 0, 0], [0, 0, -140]])
            places = 5
            self.assertAlmostEqual(map_at_random_point[0][0], 2.80609655, places=places)
            self.assertAlmostEqual(map_at_random_point[0][1], 7.37967879, places=places)

            self.assertAlmostEqual(map_at_random_point[1][0], 2.17815179, places=places)
            self.assertAlmostEqual(map_at_random_point[1][1], 9.47282782, places=places)

        # Test querying nonfinite values gives NaNs, not errors
        for itp_map in itp_maps:
            map_at_random_point = itp_map([[0, np.nan, 0], [0, 0, -140]])
            # Shape is still correct
            assert map_at_random_point.shape == (2, 2)
            # First point gives NaN
            assert np.all(np.isnan(map_at_random_point[0]))
            # Second point does not
            assert not np.any(np.isnan(map_at_random_point[1]))
