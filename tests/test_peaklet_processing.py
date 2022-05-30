import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as strat
import strax
from strax.testutils import fake_hits
import straxen
from straxen.plugins.peaklet_processing import get_tight_coin


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


def test_n_hits():
    if not straxen.utilix_is_configured():
        return
    records = np.zeros(2, dtype=strax.record_dtype())
    records['length'] = 5
    records['pulse_length'] = 5
    records['dt'] = 1
    records['channel'] = [0, 1]
    records['data'][0, :5] = [0, 1, 1, 0, 1]
    records['data'][1, :5] = [0, 1, 0, 0, 0]

    st = straxen.contexts.xenonnt_online()
    st.set_config({'hit_min_amplitude': 1})
    p = st.get_single_plugin('0', 'peaklets')
    res = p.compute(records, 0, 999)
    peaklets = res['peaklets']
    assert peaklets['n_hits'] == 3, f"Peaklet has the wrong number of hits!"


@given(fake_hits,
       strat.lists(elements=strat.integers(0, 9), min_size=20))
@settings(deadline=None)
def test_tight_coincidence(hits, channel):
    hits['area'] = 1
    hits['channel'] = channel[:len(hits)]  # In case there are less channel then hits (unlikely)
    gap_threshold = 10
    peaks = strax.find_peaks(hits,
                             adc_to_pe=np.ones(10),
                             right_extension=0, left_extension=0,
                             gap_threshold=gap_threshold,
                             min_channels=1,
                             min_area=0)

    peaks_max_time = peaks['time'] + peaks['length']//2
    hits_max_time = hits['time'] + hits['length']//2

    left = 5
    right = 5
    tight_coin_channel = get_tight_coin(hits_max_time,
                                                    hits['channel'],
                                                    peaks_max_time,
                                                    left,
                                                    right,
                                                    )
    for ind, p_max_t in enumerate(peaks_max_time):
        m_hits_in_peak = (hits_max_time >= (p_max_t - left))
        m_hits_in_peak &= (hits_max_time <= (p_max_t + right))
        n_channel = len(np.unique(hits[m_hits_in_peak]['channel']))
        assert n_channel == tight_coin_channel[ind], f'Wrong number of tight channel got {tight_coin_channel[ind]}, but expectd {n_channel}'  # noqa
