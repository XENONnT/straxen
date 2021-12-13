from straxen import InterpolatingMap, utilix_is_configured, get_resource
from unittest import TestCase, skipIf


class TestItpMaps(TestCase):
    def open_map(self, map_name, fmt):
        map_data = get_resource(map_name, fmt=fmt)
        m = InterpolatingMap(map_data, method='WeightedNearestNeighbors')
        self.assertTrue(m is not None)

    @skipIf(not utilix_is_configured(), 'Cannot download maps without db access')
    def test_lce_map(self):
        self.open_map('XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz', fmt='json.gz')
