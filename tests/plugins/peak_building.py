"""Run with python tests/plugins/peak_building.py"""
from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main
import numpy as np
import strax


@PluginTestAccumulator.register('test_area_fraction_top')
def test_area_fraction_top(self: PluginTestCase):
    merged_s2s = self.st.get_array(self.run_id, 'merged_s2s', progress_bar=False)
    _area_close_to_area_per_channel = np.isclose(
        merged_s2s['area'] / np.sum(merged_s2s['area_per_channel'], axis=1), 1)
    assert np.all(_area_close_to_area_per_channel)


@PluginTestAccumulator.register('test_saturation_correction')
def test_saturation_correction(self: PluginTestCase):
    """Manually saturate a bunch of raw-records and check that it's
    appropriately handled in the desaturation correction
    """
    st = self.st.new_context()
    rr = st.get_array(self.run_id, 'raw_records', seconds_range=(0, 10))
    assert len(rr)
    # manually saturate the data
    data = np.zeros((len(rr), len(rr['data'][0])), dtype=np.int64)
    data[:] = rr['data'].copy()
    multiply_by_factor = np.iinfo(np.int16).max / np.median(data)
    data = data * multiply_by_factor
    rr['data'] = np.clip(data, 0, np.iinfo(np.int16).max)

    pulse_proc = st.get_single_plugin(self.run_id, 'records')
    peaklet_proc = st.get_single_plugin(self.run_id, 'peaklets')
    records = pulse_proc.compute(raw_records=rr,
                                 start=np.min(rr['time']),
                                 end=np.max(strax.endtime(rr)))
    peaklets = peaklet_proc.compute(records=records['records'],
                                    start=np.min(rr['time']),
                                    end=np.max(strax.endtime(rr)))
    assert len(peaklets)
    # TODO add more tests to see if results make sense


if __name__ == '__main__':
    run_pytest_from_main()
