import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as strat

import strax
import straxen


@settings(deadline=None)
@given(strat.lists(strat.integers(min_value=0, max_value=10),
                   min_size=8, max_size=8, unique=True),
       )
def test_create_outside_peaks_region(time):
    time = np.sort(time)
    time_intervals = np.zeros(len(time)//2, strax.time_dt_fields)
    time_intervals['time'] = time[::2]
    time_intervals['length'] = time[1::2] - time[::2]
    time_intervals['dt'] = 1

    st = straxen.contexts.demo()
    p = st.get_single_plugin('0', 'peaklets')
    outside = p.create_outside_peaks_region(time_intervals, 0, np.max(time))

    touching = strax.touching_windows(outside, time_intervals, window=0)

    for tw in touching:
        print(tw)
        assert np.diff(tw) == 0, 'Intervals overlap although they should not!'
