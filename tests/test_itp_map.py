from unittest import TestCase, skipIf
from straxen import InterpolatingMap, utilix_is_configured, get_resource


class TestItpMaps(TestCase):
    def open_map(self, map_name, fmt, method='WeightedNearestNeighbors'):
        map_data = get_resource(map_name, fmt=fmt)
        m = InterpolatingMap(map_data, method=method)
        self.assertTrue(m is not None)

    @skipIf(not utilix_is_configured(), 'Cannot download maps without db access')
    def test_lce_map(self):
        self.open_map('XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz', fmt='json.gz')

    def test_array_valued(self):
        """See https://github.com/XENONnT/straxen/pull/757"""
        _map = {'coordinate_system': [[-18.3, -31.7, -111.5],
                                      [36.6, -0.0, -111.5],
                                      [-18.3, 31.7, -111.5],
                                      [-18.3, -31.7, -37.5],
                                      [36.6, -0.0, -37.5],
                                      [-18.3, 31.7, -37.5]],
                'description': 'Array_valued dummy map with lists',
                'name': 'Dummy map',
                'map': [[1.7, 11.1],
                        [1.7, 11.1],
                        [1.7, 11.0],
                        [3.3, 5.7],
                        [3.3, 5.8],
                        [3.3, 5.7]]}
        itp_map = InterpolatingMap(_map)

        # Let's do something easy, check if one fixed point yields the
        # same result if not, our interpolation map depends on the
        # straxen version?! That's bad!
        map_at_random_point = itp_map([[0, 0, 0], [0, 0, -140]])
        self.assertAlmostEqual(map_at_random_point[0][0], 2.80609655)
        self.assertAlmostEqual(map_at_random_point[1][1], 7.37967879)
