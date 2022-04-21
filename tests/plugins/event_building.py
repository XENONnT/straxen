from _core import PluginTestAccumulator, PluginTestCase
import numpy as np


@PluginTestAccumulator.register('test_exclude_s1_as_triggering_peaks_config')
def exclude_s1_as_triggering_peaks_config(self: PluginTestCase):
    """
    Test for checking the event building options, specifically the S1 (exclusion)
    """
    # Change the config to allow s1s to be triggering, also drastically
    # decrease the trigger_min_area to allow small s1s to be triggering
    st = self.st.new_context()
    st.set_config(dict(exclude_s1_as_triggering_peaks=0,
                       trigger_min_area=20,
                       ))
    events = st.get_array(self.run_id, 'event_basics')
    new_min_coincidence = int(np.max(events['s1_tight_coincidence']) - 1)

    # Create an alternative config, but with less
    st_alt = st.new_context()
    st_alt.set_config(dict(event_s1_min_coincidence=new_min_coincidence))
    events_alt = st_alt.get_array(self.run_id, 'event_basics')

    # There should be more "alt" events since we don't concatenate as
    # many peaks into one event
    self.assertGreaterEqual(len(events_alt), len(events))

    # The event durations should be smaller, since almost no s1s are
    # considered triggering so the events are extended less in the alt
    # config.
    self.assertLessEqual(
        np.mean(events_alt['endtime']-events_alt['time']),
        np.mean(events['endtime']-events['time']),

    )
