"""Run with python tests/plugins/peak_building.py."""

from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main
import numpy as np
import strax


@PluginTestAccumulator.register("test_area_fraction_top")
def test_area_fraction_top(self: PluginTestCase):
    merged_s2s = self.st.get_array(self.run_id, "merged_s2s", progress_bar=False)
    _area_close_to_area_per_channel = np.isclose(
        merged_s2s["area"] / np.sum(merged_s2s["area_per_channel"], axis=1), 1
    )
    assert np.all(_area_close_to_area_per_channel)


@PluginTestAccumulator.register("test_sum_wf")
def test_sum_wf(self: PluginTestCase):
    st_alt = self.st.new_context()
    st_alt.set_config(dict(store_data_top=True))
    peaks_alt = st_alt.get_array(self.run_id, ("peaks", "peak_basics"))
    peaks = self.st.get_array(self.run_id, ("peaks", "peak_basics"))
    np.testing.assert_array_equal(peaks_alt["data"], peaks["data"])
    # For the statement assert_array_equal seems false,
    # how can that be? The diff is <1e5 % so maybe numerical?
    np.testing.assert_array_almost_equal(peaks_alt["area_fraction_top"], peaks["area_fraction_top"])
    np.testing.assert_array_almost_equal(
        peaks["area_fraction_top"],
        np.sum(peaks["data_top"], axis=1) / np.sum(peaks["data"], axis=1),
        decimal=3,
    )


@PluginTestAccumulator.register("test_saturation_correction")
def test_saturation_correction(self: PluginTestCase):
    """Manually saturate a bunch of raw-records and check that it's appropriately handled in the
    desaturation correction."""
    st = self.st.new_context()
    st.set_config(dict(saturation_reference_length=15))
    rr = st.get_array(self.run_id, "raw_records", seconds_range=(0, 10))
    assert len(rr)
    # manually saturate the data
    data = np.zeros((len(rr), len(rr["data"][0])), dtype=np.int64)
    data[:] = rr["data"].copy()
    multiply_by_factor = np.iinfo(np.int16).max / np.median(data)
    data = data * multiply_by_factor
    rr["data"] = np.clip(data, 0, np.iinfo(np.int16).max)

    pulse_proc = st.get_single_plugin(self.run_id, "records")
    peaklet_proc = st.get_single_plugin(self.run_id, "peaklets")
    records = pulse_proc.compute(
        raw_records=rr, start=np.min(rr["time"]), end=np.max(strax.endtime(rr))
    )
    peaklets = peaklet_proc.compute(
        records=records["records"], start=np.min(rr["time"]), end=np.max(strax.endtime(rr))
    )
    assert len(peaklets)
    # TODO: add more tests to see if results make sense


@PluginTestAccumulator.register("test_tight_coincidence")
def test_tight_coincidence(self: PluginTestCase):
    """Test whether tight_coincidence is correctly reconstructed."""
    if str(self.st.key_for(self.run_id, "raw_records")) != "012882-raw_records-z7q2d2ye2t":
        print("skip checking because complexity")
        return
    peaklets = self.st.get_array(self.run_id, "peaklets", progress_bar=False)
    message = "There might be some issue in tight_coincidence."
    sum_tight_coincidence = np.sum(peaklets["tight_coincidence"])
    assert sum_tight_coincidence == 1991, message


if __name__ == "__main__":
    run_pytest_from_main()


@PluginTestAccumulator.register("test_online_peaklet_monitor")
def test_online_peaklet_monitor(self: PluginTestCase):
    """The peaklet level online monitor should conserve the peaklets it summarizes, and stay
    consistent between its counts, rates and histograms."""
    monitor = self.st.get_array(self.run_id, "online_peaklet_monitor", progress_bar=False)
    peaklets = self.st.get_array(self.run_id, "peaklets", progress_bar=False)
    assert len(monitor), "No online peaklet monitor data"

    # Every peaklet is counted exactly once
    assert monitor["n_peaklets"].sum() == len(peaklets)

    # Single electrons are a subset of the S2-like peaklets
    assert np.all(monitor["n_se"] <= monitor["n_s2_peaklets"])
    assert np.all(monitor["n_s2_peaklets"] <= monitor["n_peaklets"])

    # Histograms cannot hold more entries than there are peaklets to fill them
    assert np.all(monitor["se_area_hist"].sum(axis=1) <= monitor["n_s2_peaklets"])
    assert np.all(monitor["se_aft_hist"].sum(axis=1) <= monitor["n_se"])
    assert np.all(monitor["area_vs_width_hist"].sum(axis=(1, 2)) <= monitor["n_peaklets"])

    # Rates are counts per unit of livetime
    livetime = (monitor["endtime"] - monitor["time"]) / 1e9
    assert np.all(livetime > 0)
    np.testing.assert_array_almost_equal(
        monitor["peaklet_rate"], monitor["n_peaklets"] / livetime, decimal=3
    )
    np.testing.assert_array_almost_equal(monitor["se_rate"], monitor["n_se"] / livetime, decimal=3)

    # The gain estimate is either inside the single-electron window or not set
    se_low, se_high = self.st.config.get("se_monitor_window", (15.0, 70.0))
    is_set = monitor["se_gain"] > 0
    assert np.all(monitor["se_gain"][is_set] > se_low)
    assert np.all(monitor["se_gain"][is_set] < se_high)


@PluginTestAccumulator.register("test_online_peaklet_monitor_is_bounded")
def test_online_peaklet_monitor_is_bounded(self: PluginTestCase):
    """The number of rows per chunk must stay below the configured maximum, whatever the time bin,
    since one chunk becomes one document of the online monitor database."""
    max_rows = 3
    st = self.st.new_context()
    st.set_config(dict(peaklet_monitor_time_bin=int(1e6), peaklet_monitor_max_rows=max_rows))
    for chunk in st.get_iter(self.run_id, "online_peaklet_monitor", progress_bar=False):
        assert len(chunk.data) <= max_rows
