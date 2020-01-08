import straxen
import numpy as np
from hypothesis import strategies, given


@given(strategies.lists(strategies.integers(0, 1), min_size=2, max_size=30))
def test_start_end_merge(merge_with_next):
    merge_with_next = np.array(merge_with_next, dtype=np.int8)
    y = merge_with_next

    try:
        start_merge, end_merge = straxen.get_start_end(merge_with_next)
    except AssertionError as e:
        # We want to have an assertion error in this case since one is trying to
        # merge the last peak to another peak (but there is not later than last)
        if len(y) and y[-1] == 1:
            return
        # In other cases, assertion errors will be re-raised
        else:
            raise e
    assert len(start_merge) == len(end_merge)

    if y[0]:
        # We must start at index 0 if merge_index[0] == 1
        assert 0 in start_merge
    else:
        assert 0 not in start_merge

    for i in range(1, len(y)):
        if y[i] < y[i-1]:
            assert i + 1 in end_merge, f"{i} not in end merge"
        elif y[i] > y[i-1]:
            assert i in start_merge, f"{i} not in start merge"
        else:
            assert (i + 1 not in end_merge) and (i not in start_merge), f"Fail at {i}"

