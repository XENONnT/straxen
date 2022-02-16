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
                              channel_range=(2000, 2119),
                              length_range=(1, 1), ):
    """
    Function which generates a hypothesis strategy for a fixed number
    of disjoint intervals

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

    Note:
        You can use create_disjoint_intervals().example() to see an
        example.
        If you do not want to specify the bounds for any of the "_range"
        parameters set the corresponding bound to None.

        Somehow hypothesis complains that the creation of these events
        takes too long ~2 s for 50 intervals. You can disable the
        corresponding healt checks via:" @settings(
        suppress_health_check=[hypothesis.HealthCheck.large_base_example,
         hypothesis.HealthCheck.too_slow])"
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


@settings(suppress_health_check=[hypothesis.HealthCheck.large_base_example,
                                 hypothesis.HealthCheck.too_slow],
          deadline=None)
@given(create_disjoint_intervals(strax.hitlet_dtype(),
                                 n_intervals=7,
                                 dt=1,
                                 time_range=(0, 15),
                                 channel_range=(2000, 2010),
                                 length_range=(1, 4), ),
       hst.integers(1, 3),
       )
def test_nveto_event_building(hitlets,
                              coincidence):
    """
    In this test we test the code of
    straxen.plugins.veto_evnets.find_veto_events
    """
    hitlets = strax.sort_by_time(hitlets)

    event_intervals = straxen.plugins.nveto_recorder.find_coincidence(hitlets,
                                                                      coincidence,
                                                                      300)

    mes = 'Found overlapping events returned by "coincidence".'
    assert np.all(event_intervals['endtime'][:-1] - event_intervals['time'][1:] < 0), mes

    # Get hits which touch the event window, this can lead to ambiguities
    # which we will solve subsequently.
    hitlets_ids_in_event = strax.touching_windows(hitlets, event_intervals)
    # First check for empty events, since ambiguity check will merge intervals:
    mes = f'Found an empty event without any hitlets: {hitlets_ids_in_event}.'
    assert np.all(np.diff(hitlets_ids_in_event) != 0), mes

    # Solve ambiguities (merge overlapping intervals)
    interval_truth = _test_ambiguity(hitlets_ids_in_event)
    hitlets_ids_in_event = straxen.plugins.veto_events._solve_ambiguity(hitlets_ids_in_event)

    mes = f'Found ambigious event for {hitlets_ids_in_event} with turth {interval_truth}'
    assert np.all(hitlets_ids_in_event == interval_truth), mes

    # Check if events satisfy the coincidence requirement:
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


@settings(suppress_health_check=[hypothesis.HealthCheck.large_base_example,
                                 hypothesis.HealthCheck.too_slow],
          deadline=None)
@given(create_disjoint_intervals(strax.hitlet_dtype(),
                                 n_intervals=7,
                                 dt=1,
                                 time_range=(0, 15),
                                 channel_range=(2000, 2010),
                                 length_range=(1, 4), ),
       hnp.arrays(np.float32, elements=hst.floats(-1, 10, width=32), shape=7),
       )
def test_nveto_event_plugin(hitlets, area):
    hitlets['area'] = area
    hitlets = strax.sort_by_time(hitlets)
    events, hitlets_ids_in_event = straxen.find_veto_events(hitlets,
                                                            3,
                                                            300,
                                                            0)

    straxen.plugins.veto_events.compute_nveto_event_properties(events,
                                                               hitlets,
                                                               hitlets_ids_in_event,
                                                               start_channel=2000)
    # Test some of the parameters:
    for e, hit_ids in zip(events, hitlets_ids_in_event):
        hits = hitlets[hit_ids[0]:hit_ids[1]]

        assert e['time'] == np.min(hits['time']), f'Event start is wrong (hit_ids: hit_ids)'
        assert e['endtime'] == np.max(strax.endtime(hits)), f'Event end is wrong (hit_ids: hit_ids)'
        assert np.isclose(e['area'], np.sum(hits['area'])), f'Event area is wrong for {e["area"]}, {hits["area"]}'
        mes = f'Event n_contributing_pmt is wrong for {e["n_contributing_pmt"]}, {hits["channel"]}'
        assert e['n_contributing_pmt'] == len(np.unique(hits['channel'])), mes
        assert e['n_hits'] == len(hits), f'Event n_hits is wrong for {e["n_hits"]}, {hits}'

    # -----------------------
    # Check if updated events
    # have the correct boundaries:
    # -----------------------
    if len(events) > 1:
        mes = f'Updated event boundaries overlap! {events}'
        assert (events['endtime'][:-1] - events['time'][1:]) < 0, mes

    split_hitlets = strax.split_by_containment(hitlets, events)

    for sbc_hitlets, tw_hitlet_id in zip(split_hitlets, hitlets_ids_in_event):
        h = hitlets[tw_hitlet_id[0]:tw_hitlet_id[1]]
        mes = ('Touching windows and split_by_containment yield different hitlets'
               ' after updating the event boundaries. This should not have happened.')
        assert np.all(sbc_hitlets == h), mes

    # Test event positions:
    try:
        npmt_pos = straxen.get_resource('nveto_pmt_position.csv', fmt='csv')
        npmt_pos = npmt_pos.to_records(index=False)
    except FileNotFoundError:
        npmt_pos = np.ones(120, dtype=[('x', np.float64),
                                       ('y', np.float64),
                                       ('z', np.float64)])

    events_angle = np.zeros(len(events),
                            dtype=straxen.plugins.veto_events.veto_event_positions_dtype())

    straxen.plugins.veto_events.compute_positions(events_angle,
                                                  events,
                                                  split_hitlets,
                                                  npmt_pos,
                                                  start_channel=2000)

    angle = straxen.plugins.veto_events.get_average_angle(split_hitlets,
                                                          npmt_pos,
                                                          start_channel=2000)
    # Compute truth angles:
    truth_angle = np.angle(events_angle['pos_x']+events_angle['pos_y']*1j)
    # Replace not defined angles, into zeros to match np.angles return
    # and to simplify comparison
    m = (events_angle['pos_x'] == 0) & (events_angle['pos_y'] == 0)
    angle[m] = 0

    # Fixing +2pi issue and np.angle [-180, 180] and [0, 360) convention
    # issue.
    angle = angle % (2*np.pi)
    truth_angle = truth_angle % (2*np.pi)

    # Sometimes it may happen due to numerical precision that one angle is slightly
    # larger than 2 pi while the other is slightly smaller. In that case we have to
    # fix it:
    if np.isclose(angle, 2*np.pi):
        angle -= 2*np.pi
    if np.isclose(truth_angle, 2*np.pi):
        truth_angle -= 2*np.pi

    # Compare angle, also indirectly tests average x/y/z
    mes = f'Event angle did not match expected {truth_angle}, got {angle}.'
    assert np.isclose(angle, truth_angle), mes
