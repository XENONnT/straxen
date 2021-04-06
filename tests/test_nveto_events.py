import strax
import straxen

import numpy as np

import hypothesis
from hypothesis import given, settings
import hypothesis.strategies as hst
import hypothesis.extra.numpy as hnp


@hst.composite
def create_disjoint_intervals(draw,
                              dtype,
                              n_intervals=10,
                              dt=1,
                              time_range=(0, 100),
                              channel_range=(2000, 2120),
                              length_range=(1, 1), ):
    """
    Function which generates a hypothesis strategy for a fixed number
    of disjoint intervals

    Note:
        You can use create_disjoint_intervals().example() to see an
        example.
        If you do not want to specify the bounds for any of the "_range"
        parameters set the corresponding bound to None.

    :param dtype: Can be any strax-like dtype either with endtime or
        dt and length field.
    :param n_intervals: How many disjoint intervals should be returned.
    :param dt: Sampling field, only needed for length + dt fields.
    :param time_range: Time range in which random numbers will be
        generated.
    :param channel_range: Range of channels for which the disjoint
        intervals will be generated. For a single channel set min/max
        equal.
    :param length_range: Range how long time intervals can be.
    :return: hypothesis strategy which can be used in @given
    """
    n = 0

    if not hasattr(dtype, 'fields'):
        # Convert dtype into numpy dtype
        dtype = np.dtype(dtype)

    is_dt = True
    if 'endtime' in dtype.fields:
        # Check whether interval uses dt fields or endtime
        is_dt = False

    stratgey_example = np.zeros(n_intervals, dtype)
    if is_dt:
        stratgey_example['dt'] = dt

    while n < n_intervals:
        # Create interval values:
        time = draw(hst.integers(*time_range))
        channel = draw(hst.integers(*channel_range))
        length = draw(hst.integers(*length_range))

        # Check if objects are disjoint:
        if _test_disjoint(stratgey_example[:n], time, length, channel, dt):
            stratgey_example[n]['time'] = time
            stratgey_example[n]['channel'] = channel
            if is_dt:
                stratgey_example[n]['length'] = length
            else:
                stratgey_example[n]['endtime'] = time + int(length * dt)
            n += 1
    return stratgey_example


def _test_disjoint(intervals, time, length, channel, dt):
    int_ch = intervals[intervals['channel'] == channel]
    if not len(int_ch):
        # No interval in the given channel yet
        return True

    endtime = strax.endtime(int_ch)
    m = (int_ch['time'] <= time) & (time < endtime)
    edt = time + length * dt
    m |= (int_ch['time'] <= edt) & (edt < endtime)
    if np.any(m):
        # Found an overlap:
        return False
    else:
        return True


@settings(suppress_health_check=[hypothesis.HealthCheck.large_base_example, hypothesis.HealthCheck.too_slow])
@given(create_disjoint_intervals(strax.hitlet_dtype(),
                                 n_intervals=50,
                                 dt=1,
                                 time_range=(0, 1000),
                                 channel_range=(2000, 2120),
                                 length_range=(20, 80), ), hst.integers(1, 3))
def test_nveto_events_and_coincidence(hitlets, coincidence):
    hitlets = strax.sort_by_time(hitlets)

    intervals = straxen.plugins.nveto_recorder.coincidence(hitlets, coincidence, 50)
    events = np.zeros(len(intervals), strax.time_fields)
    events['time'] = intervals[:, 0]
    events['endtime'] = intervals[:, 1]

    mes = 'Found overlapping events returned by "coincidence".'
    assert np.all(events['endtime'][:-1] - events['time'][1:] < 0), mes

    # Get hits which touch the event window, this can lead to ambiguities
    # which we will solve subsequently.
    hitlets_ids_in_event = strax.touching_windows(hitlets, events)
    print(hitlets_ids_in_event)
    # First check for empty events, since amiguity check will merge intervals:
    mes =  f'Found an empty event without any hitlets: {hitlets_ids_in_event}.'
    assert np.all(np.diff(hitlets_ids_in_event) != 0), mes

    # Solve ambiguities (merge overlapping intervals) remove empty once and
    # check.
    interval_truth = _test_ambiguity(hitlets_ids_in_event)

    hitlets_ids_in_event = straxen.plugins.veto_events.solve_ambiguity(hitlets_ids_in_event)
    m = np.diff(hitlets_ids_in_event).flatten() != 0
    hitlets_ids_in_event = hitlets_ids_in_event[m]

    mes = f'Found ambigious event for {hitlets_ids_in_event} with turth {interval_truth}'
    assert np.all(hitlets_ids_in_event == interval_truth), mes

    # Check if events satisfy the interval requirement:
    mes = f'Found an event with less than 3 hitelts. {hitlets_ids_in_event}'
    assert np.all(np.diff(hitlets_ids_in_event) >= coincidence), mes


def _test_ambiguity(hitlets_ids_in_event):
    """
    Returns overlap free intervals for ambiguity check.
    """
    res = []
    start, end = hitlets_ids_in_event[0]
    for ids in hitlets_ids_in_event[1:]:
        s, e = ids
        if s < end:
            # Overlapping event:
            end = e
        else:
            # New event:
            res.append([start, end])
            start = s
            end = e
    # Add last interval:
    res.append([start, end])
    return res