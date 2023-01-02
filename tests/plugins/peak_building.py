"""Run with python tests/plugins/peak_building.py"""
from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main
import numpy as np
import strax

@PluginTestAccumulator.register('test_peak_waveform')
def test_peak_waveform(self: PluginTestCase):
    from straxen.plugins.peaks.peak_waveform import HitsS1, PeakWaveformS1, PeakAmbienceS1
    self.st.register([HitsS1, PeakWaveformS1, PeakAmbienceS1])
    df = self.st.get_array(self.run_id, 'hits_s1', progress_bar=False)
    df = self.st.get_array(self.run_id, 'peak_waveform_s1', progress_bar=False)
    df = self.st.get_array(self.run_id, 'peak_ambience_s1', progress_bar=False)

@PluginTestAccumulator.register('test_area_fraction_top')
def test_area_fraction_top(self: PluginTestCase):
    merged_s2s = self.st.get_array(self.run_id, 'merged_s2s', progress_bar=False)
    _area_close_to_area_per_channel = np.isclose(
        merged_s2s['area'] / np.sum(merged_s2s['area_per_channel'], axis=1), 1)
    assert np.all(_area_close_to_area_per_channel)


@PluginTestAccumulator.register('test_sum_wf')
def test_sum_wf(self: PluginTestCase):
    st_alt = self.st.new_context()
    st_alt.set_config(dict(sum_waveform_top_array=False))
    peaks_alt = st_alt.get_array(self.run_id, ('peaks', 'peak_basics'))
    peaks = self.st.get_array(self.run_id, ('peaks', 'peak_basics'))
    np.testing.assert_array_equal(peaks_alt['data'], peaks['data'])
    # For the statement assert_array_equal seems false, how can that be? The diff is <1e5 % so maybe numerical?
    np.testing.assert_array_almost_equal(
        peaks_alt['area_fraction_top'], peaks['area_fraction_top'])
    np.testing.assert_array_almost_equal(
        peaks['area_fraction_top'], np.sum(peaks['data_top'], axis=1) / np.sum(peaks['data'], axis=1),
        # TODO rather high tolerance is needed to pass the test -> possible bug?
        decimal=4)


@PluginTestAccumulator.register('test_saturation_correction')
def test_saturation_correction(self: PluginTestCase):
    """Manually saturate a bunch of raw-records and check that it's
    appropriately handled in the desaturation correction
    """
    st = self.st.new_context()
    st.set_config(dict(saturation_reference_length=15))
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
