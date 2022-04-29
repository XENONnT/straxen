from _core import PluginTestAccumulator, PluginTestCase
import numpy as np
import straxen


@PluginTestAccumulator.register('test_exclude_s1_as_triggering_peaks_config')
def exclude_s1_as_triggering_peaks_config(self: PluginTestCase, trigger_min_area=10):
    """
    Test for checking the event building options, specifically the S1 (exclusion)
    """
    # Change the config to allow s1s to be triggering, also drastically
    # decrease the trigger_min_area to allow small s1s to be triggering
    st = self.st.new_context()
    st.set_config(dict(exclude_s1_as_triggering_peaks=0,
                       trigger_min_area=trigger_min_area,
                       ))
    events = st.get_array(self.run_id, 'event_basics', progress_bar=False)
    new_min_coincidence = int(np.max(events['s1_tight_coincidence']) - 1)

    # Create an alternative config, but with less
    st_alt = st.new_context()
    st_alt.set_config(dict(event_s1_min_coincidence=new_min_coincidence))
    events_alt = st_alt.get_array(self.run_id, 'event_basics', progress_bar=False)

    # The event durations should be smaller, since almost no s1s are
    # considered triggering so the events are extended less in the alt
    # config.
    self.assertLessEqual(
        np.mean(events_alt['endtime'] - events_alt['time']),
        np.mean(events['endtime'] - events['time']),
    )

    # Check the triggers of the alternate config
    event_plugin = st_alt.get_single_plugin(self.run_id, 'events')
    triggers = get_triggering_peaks(events_alt,
                                    event_plugin.left_extension,
                                    event_plugin.right_extension)
    s1_triggers = triggers[triggers['type'] == 1]
    self.assertTrue(all(s1_triggers['tight_coincidence'] >= new_min_coincidence))
    self.assertTrue(all(triggers['area'] >= trigger_min_area))


@PluginTestAccumulator.register('test_event_info')
def test_event_info(self):
    """Do a dummy check on event-info that it loads"""
    df = self.st.get_df(self.run_id, 'event_info')

    assert len(df) > 0
    assert 'cs1' in df.columns
    assert df['cs1'].sum() > 0
    assert not np.all(np.isnan(df['x'].values))


@PluginTestAccumulator.register('test_event_info_double')
def test_event_info_double(self):
    """Do a dummy check on event-info that it loads"""
    df = self.st.get_df(self.run_id, 'event_info_double')
    assert 'cs2_a' in df.columns
    assert df['cs2_a'].sum() > 0
    assert len(df) > 0


@PluginTestAccumulator.register('test_get_livetime_sec')
def test_get_livetime_sec(self):
    st = self.st
    events = st.get_array(self.run_id, 'events')
    straxen.get_livetime_sec(st, self.run_id, things=events)


def get_triggering_peaks(events, left_extension, right_extension):
    """
    Extract the first and last triggering peaks from an event and return type, area an tight_coincidence
    """
    peaks = np.zeros(len(events) * 4,
                     dtype=[(('type', 'type of peak'), np.int8),
                            (('tight_coincidence', 'tc level'), np.int16,),
                            (('area', 'peak area'), np.float32),
                            ])
    peaks_seen = 0
    for event in events:
        for peak in 's1_ s2_ alt_s1_ alt_s2_'.split():
            # We know that the event boundaries are set by the first/last
            # triggering peaks, so just select those peaks that have defined the
            # boundary (if we can still). This is not a complete list of
            # triggers, but works well enough for this test
            is_first_trigger = event[f'{peak}time'] - left_extension == event['time']
            is_last_trigger = event[f'{peak}endtime'] + right_extension == event['endtime']
            if is_first_trigger or is_last_trigger:
                peaks[peaks_seen]['type'] = int(peak[-2])
                for field in 'area tight_coincidence'.split():
                    peaks[peaks_seen][field] = event[f'{peak}{field}']
                peaks_seen += 1
    return peaks[:peaks_seen]
