from _core import PluginTestAccumulator
from unittest import TestCase


@PluginTestAccumulator.register('test_exclude_s1_as_triggering_peaks_config')
def exclude_s1_as_triggering_peaks_config(self: TestCase):
    st = self.st

    # Create an alternative config with almost identical settings
    st_alt = st.new_context()
    st_alt.set_config(dict(exclude_s1_as_triggering_peaks=0,
                           event_s1_min_coincidence=2,
                           ),
                      )
    events = st.get_array(self.run_id, 'event_basics')
    events_alt = st_alt.get_array(self.run_id, 'event_basics')

    self.assertAlmostEqual(len(events), len(events_alt))
