from hypothesis import given, settings
import strax
import strax.testutils
import straxen


@settings(deadline=None)
@given(strax.testutils.several_fake_records)
def test_count_pulses(records):
    _check_pulse_count(records)


@settings(deadline=None)
@given(strax.testutils.several_fake_records_one_channel)
def test_count_pulses_2(records):
    _check_pulse_count(records)


def _check_pulse_count(records):
    # TODO: numba starts complaining if n_channels == 1, maybe file bug?
    n_ch = records['channel'].max() + 2 if len(records) else 0
    counts = straxen.plugins.pulse_processing.count_pulses(
        records, n_channels=n_ch)

    assert counts.dtype == straxen.pulse_count_dtype(n_ch)

    # TODO temporary hack until we fix strax issue #239
    if not len(records):
        assert len(counts) == 0
        return

    assert len(counts) == 1
    count = counts[0]

    # Check total pulse count and area
    for ch, n in enumerate(counts[0]['pulse_count']):
        rc = records[records['channel'] == ch]
        rc0 = rc[rc['record_i'] == 0]

        assert n == len(rc0)
        assert count['pulse_area'][ch] == rc['area'].sum()

        # Not sure how to check lone pulses other than duplicating logic
        # already in count_pulses, so just do a basic check:
        assert count['lone_pulse_area'][ch] <= count['pulse_area'][ch]
