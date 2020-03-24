import numpy as np
import hypothesis

import strax.testutils
import straxen


def channel_split_naive(r, channel_ranges):
    """Slower but simpler implementation of straxen.split_channel_ranges"""
    results = []
    for left, right in channel_ranges:
        results.append(r[np.in1d(r['channel'], np.arange(left, right + 1))])
    return results


@hypothesis.settings(deadline=None)
@hypothesis.given(strax.testutils.several_fake_records)
def test_channel_split(records):
    channel_range = np.asarray([[0, 0], [1, 2], [3, 3], [4, 999]])
    result = list(straxen.split_channel_ranges(records, channel_range))
    result_2 = channel_split_naive(records, channel_range)

    assert len(result) == len(result_2)
    for i in range(len(result)):
        np.testing.assert_array_equal(
            np.unique(result[i]['channel']),
            np.unique(result_2[i]['channel']))
        np.testing.assert_array_equal(result[i], result_2[i])