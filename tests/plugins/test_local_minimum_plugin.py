import straxen
import numpy as np
from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main
from straxen.plugins.events import local_minimum_info


@PluginTestAccumulator.register('test_event_local_min_info')
def test_event_local_min_info(self: PluginTestCase):
    """Try making local minimum info"""
    st = self.st
    st.register(local_minimum_info.LocalMinimumInfo)
    st.make(self.run_id, 'event_local_min_info')


@PluginTestAccumulator.register('test_identify_local_extrema')
def test_identify_local_extrema(self: PluginTestCase):

    """
    Tests whether the local extrema are identified properly
    """
    test_peak_1, test_peak_2, _ = _get_test_peaks()
    max1, min1 = local_minimum_info.identify_local_extrema(test_peak_1)
    max2, min2 = local_minimum_info.identify_local_extrema(test_peak_2)

    assert np.all(min1 == min2)
    assert len(max2)-len(min2) == 1
    assert len(max1)-len(min1) == 1
    assert min1[0]==100
    assert len(max1) == 2
    assert len(max2) == 2


@PluginTestAccumulator.register('test_full_gap_percent_valley')
def test_full_gap_percent_valley(self: PluginTestCase):

    """
    Tests if the gaps and valleys are identified properly
    """
    test_peak_1, test_peak_2, test_peak_3 = _get_test_peaks()
    max1, min1 = local_minimum_info.identify_local_extrema(test_peak_1)
    max2, min2 = local_minimum_info.identify_local_extrema(test_peak_2)
    max3, min3 = local_minimum_info.identify_local_extrema(test_peak_3)

    valley_gap1, valley1 = local_minimum_info.full_gap_percent_valley(test_peak_1, max1, min1, 0.9, 1)
    valley_gap2, valley2 = local_minimum_info.full_gap_percent_valley(test_peak_2, max2, min2, 0.9, 1)
    valley_gap3, valley3 = local_minimum_info.full_gap_percent_valley(test_peak_3, max3, min3, 0.9, 1)

    #peak 2 should have the shortest gap and the shortest valley
    #peak 3 should have the deepest valley
    #peak 3 should have the widest gap

    assert (valley_gap2 < valley_gap1) & (valley_gap2 < valley_gap3)
    assert (valley2 < valley1) & (valley2 < valley3)
    assert (valley3 > valley1) & (valley3 > valley2)
    assert (valley_gap3 > valley_gap1) & (valley_gap3 > valley_gap2)

def _get_test_peaks():
    #A fake peak to test the functions on
    test_time = np.arange(200)
    std1, std2 = 15, 20

    #peak 1: relatively narrow peaks, separated closely
    #peak 2: relatively wide peaks, separated closely
    #peak 3: relatively wide peaks, separated far
    test_peak_1 = np.exp(-(test_time-75)**2/(2*std1**2))+np.exp(-(test_time-125)**2/(2*std1**2))
    test_peak_2 = np.exp(-(test_time-75)**2/(2*std2**2))+np.exp(-(test_time-125)**2/(2*std2**2))
    test_peak_3 = np.exp(-(test_time-50)**2/(2*std2**2))+np.exp(-(test_time-150)**2/(2*std2**2))
    return test_peak_1, test_peak_2, test_peak_3

if __name__ == '__main__':
    run_pytest_from_main()
