"""Run with python tests/plugins/event_building.py."""

import os
import tempfile

import strax

from _core import PluginTestAccumulator, PluginTestCase, run_pytest_from_main
import numpy as np
import straxen
import pandas as pd
from datetime import datetime


@PluginTestAccumulator.register("test_exclude_s1_as_triggering_peaks_config")
def exclude_s1_as_triggering_peaks_config(self: PluginTestCase, trigger_min_area=10):
    """Test for checking the event building options, specifically the S1 (exclusion)"""
    if _is_empty_data_test(self.st, self.run_id):
        return
    # Change the config to allow s1s to be triggering, also drastically
    # decrease the trigger_min_area to allow small s1s to be triggering
    st = self.st.new_context()
    st.set_config(
        dict(
            exclude_s1_as_triggering_peaks=0,
            trigger_min_area=trigger_min_area,
        )
    )
    events = st.get_array(self.run_id, "event_basics", progress_bar=False)
    new_min_coincidence = int(np.max(events["s1_tight_coincidence"]) - 1)

    # Create an alternative config, but with less
    st_alt = st.new_context()
    st_alt.set_config(dict(event_s1_min_coincidence=new_min_coincidence))
    events_alt = st_alt.get_array(self.run_id, "event_basics", progress_bar=False)

    # The event durations should be smaller, since almost no s1s are
    # considered triggering so the events are extended less in the alt
    # config.
    self.assertLessEqual(
        np.mean(events_alt["endtime"] - events_alt["time"]),
        np.mean(events["endtime"] - events["time"]),
    )

    # Check the triggers of the alternate config
    event_plugin = st_alt.get_single_plugin(self.run_id, "events")
    triggers = get_triggering_peaks(
        events_alt, event_plugin.left_extension, event_plugin.right_extension
    )
    s1_triggers = triggers[triggers["type"] == 1]
    self.assertTrue(all(s1_triggers["tight_coincidence"] >= new_min_coincidence))
    self.assertTrue(all(triggers["area"] >= trigger_min_area))


@PluginTestAccumulator.register("test_event_info")
def test_event_info(self):
    """Do a dummy check on event-info that it loads."""
    if _is_empty_data_test(self.st, self.run_id):
        return
    df = self.st.get_df(self.run_id, "event_info")

    assert len(df) > 0, self.st.key_for(self.run_id, "raw_records")
    assert "cs1" in df.columns
    assert df["cs1"].sum() > 0
    assert not np.all(np.isnan(df["x"].values))


@PluginTestAccumulator.register("test_event_info_double")
def test_event_info_double(self):
    """Do a dummy check on event-info that it loads."""
    if _is_empty_data_test(self.st, self.run_id):
        return
    df = self.st.get_df(self.run_id, "event_info_double")
    assert "cs2_a" in df.columns
    assert df["cs2_a"].sum() > 0
    assert len(df) > 0


@PluginTestAccumulator.register("test_event_ms_naive")
def test_event_ms_naive(self):
    """Do a dummy check on event-info that it loads."""
    if _is_empty_data_test(self.st, self.run_id):
        return
    df = self.st.get_df(self.run_id, targets=("event_info", "event_ms_naive"))
    assert "cs2_sum" in df.columns
    assert "cs2_wo_timecorr_sum" in df.columns
    assert "cs2_wo_elifecorr_sum" in df.columns
    assert "cs2_area_fraction_top_avg" in df.columns
    assert df["cs2_sum"].sum() > 0
    assert df["cs2_wo_timecorr_sum"].sum() > 0
    assert df["cs2_wo_elifecorr_sum"].sum() > 0
    assert df["cs2_area_fraction_top_avg"].sum() > 0


@PluginTestAccumulator.register("test_get_livetime_sec")
def test_get_livetime_sec(self):
    st = self.st
    events = st.get_array(self.run_id, "events")
    if len(events):
        straxen.get_livetime_sec(st, self.run_id, things=events)


@PluginTestAccumulator.register("test_event_info_double_w_double_peaks")
def test_event_info_double_w_double_peaks(self: PluginTestCase, trigger_min_area=10):
    """Try building event-info double with very long events."""
    st = self.st.new_context()
    ev = st.get_array(self.run_id, "events")
    if not len(ev):
        return
    ev_time_diff = np.median(np.diff(ev["time"]))
    # increase the event_extension such that we start merging several events
    st.set_config(dict(event_right_extension=ev_time_diff))
    st.get_array(self.run_id, "event_info_double")

    distinct_channels = st.get_single_plugin(self.run_id, "distinct_channels")
    events = st.get_array(self.run_id, "event_basics")
    # Make alt == main just to test that we are able to compute that
    # all have no distinct channels
    events["alt_s1_index"] = events["s1_index"]
    peaks = st.get_array(self.run_id, "peaks")
    split_peaks = strax.split_by_containment(peaks, events)
    for event, split_peak in zip(events, split_peaks):
        res = distinct_channels.compute_loop(event, split_peak)
        assert res["alt_s1_distinct_channels"] == 0


def get_triggering_peaks(events, left_extension, right_extension):
    """Extract the first and last triggering peaks from an event and return type, area an
    tight_coincidence."""
    peaks = np.zeros(
        len(events) * 4,
        dtype=[
            (("type", "type of peak"), np.int8),
            (
                ("tight_coincidence", "tc level"),
                np.int16,
            ),
            (("area", "peak area"), np.float32),
        ],
    )
    peaks_seen = 0
    for event in events:
        for peak in "s1_ s2_ alt_s1_ alt_s2_".split():
            # We know that the event boundaries are set by the first/last
            # triggering peaks, so just select those peaks that have defined the
            # boundary (if we can still). This is not a complete list of
            # triggers, but works well enough for this test
            is_first_trigger = event[f"{peak}time"] - left_extension == event["time"]
            is_last_trigger = event[f"{peak}endtime"] + right_extension == event["endtime"]
            if is_first_trigger or is_last_trigger:
                peaks[peaks_seen]["type"] = int(peak[-2])
                for field in "area tight_coincidence".split():
                    peaks[peaks_seen][field] = event[f"{peak}{field}"]
                peaks_seen += 1
    return peaks[:peaks_seen]


@PluginTestAccumulator.register("test_partitioned_tpc_corrected_areas")
def test_corrected_areas(self: PluginTestCase, ab_value=20, cd_value=21):
    """Run the test in ../test_url_config.TestURLConfig.test_seg_file_json on corrected_areas."""
    fake_file = {
        "time": [
            datetime(2000, 1, 1).timestamp() * 1e9,
            datetime(2021, 1, 1).timestamp() * 1e9,
            datetime(2040, 1, 1).timestamp() * 1e9,
        ],
        "ab": [10, ab_value, 30],
        "cd": [11, cd_value, 31],
    }
    # This example also works well with dataframes!
    temp_dir = tempfile.TemporaryDirectory()
    fake_file_name = os.path.join(temp_dir.name, "test_seg.csv")
    pd.DataFrame(fake_file).to_csv(fake_file_name)

    self.st.set_config({
        "se_gain": (
            f"itp_dict://resource://{fake_file_name}?run_id=plugin.run_id&fmt=csv&itp_keys=ab,cd"
        )
    })
    # Try loading some new data with the interpolated dictionary
    _ = self.st.get_array(self.run_id, "corrected_areas")
    temp_dir.cleanup()


def _is_empty_data_test(st, run_id):
    return str(st.key_for(run_id, "raw_records")) == f"{run_id}-raw_records-5uvrrzwhnl"


@PluginTestAccumulator.register("test_alternative_s2_areas")
def test_alternative_s2_areas(self: PluginTestCase, trigger_min_area=10):
    """Test if the extracted S2 peaks in an event are the largest peaks in the event time range if
    both force_main_before_alt and force_alt_s2_in_max_drift_time are True."""

    # Initialize plugin and config options
    st = self.st.new_context()
    st.set_config({"force_main_before_alt": True, "force_alt_s2_in_max_drift_time": True})
    run_id = "000000"
    event_basics_plugin = st.get_single_plugin(run_id, "event_basics")

    # Build fake two events and a few peaks, one event has an S1, one event does not.
    dummy_events = np.zeros(2, dtype=event_basics_plugin.dtype)
    dummy_events["time"] = np.array([0, 105])
    dummy_events["endtime"] = np.array([100, 205])

    # Prepare some dummy peaks
    peaks_dtype = strax.merged_dtype((
        st.get_single_plugin(run_id, "peak_basics").dtype,
        st.get_single_plugin(run_id, "peak_proximity").dtype,
        st.get_single_plugin(run_id, "peak_positions").dtype,
    ))

    peaks_for_event_1 = np.zeros(5, dtype=peaks_dtype)
    peaks_for_event_1["time"] = np.arange(10, 105, 20)
    peaks_for_event_1["center_time"] = peaks_for_event_1["time"] + 1
    peaks_for_event_1["endtime"] = peaks_for_event_1["time"] + 2
    peaks_for_event_1["type"] = 2
    peaks_for_event_1["tight_coincidence"] = 5
    peaks_for_event_1["area"] = np.array([10, 20, 70, 100, 50])

    peaks_for_event_2 = np.zeros(5, dtype=peaks_dtype)
    peaks_for_event_2["time"] = np.arange(110, 205, 20)
    peaks_for_event_2["center_time"] = peaks_for_event_2["time"] + 1
    peaks_for_event_2["endtime"] = peaks_for_event_2["time"] + 2
    peaks_for_event_2["type"] = 2
    peaks_for_event_2["type"][1] = 1
    peaks_for_event_2["tight_coincidence"] = 5
    peaks_for_event_2["area"] = np.array([10, 20, 70, 100, 50])

    result = event_basics_plugin.compute(
        dummy_events, np.concatenate((peaks_for_event_1, peaks_for_event_2))
    )

    # Check for both events if the peak that was identified as
    # main s2 really is the (second-)largest s2 peak in that time range
    assert (result[0]["s2_area"] == 70) & (result[0]["alt_s2_area"] == 100)
    assert (result[1]["s2_area"] == 70) & (result[1]["alt_s2_area"] == 100)


if __name__ == "__main__":
    run_pytest_from_main()
