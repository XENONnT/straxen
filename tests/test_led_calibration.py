import strax
from strax.testutils import several_fake_records
import straxen
import numpy as np
from hypothesis import given, settings


@given(several_fake_records)
@settings(deadline=None)
def test_ext_timings_nv(records):
    """
    Little test for nVetoExtTimings.
    """
    records['pulse_length'] = records['length']

    st = straxen.contexts.xenonnt_led()
    plugin = st.get_single_plugin('1', 'ext_timings_nv')
    hits = strax.find_hits(records)
    result = plugin.compute(hits, records)

    assert len(result) == len(hits)
    assert np.all(result['time'] == hits['time'])
    assert np.all(result['pulse_i'] == hits['record_i'])
    true_dt = hits['time'] - records[hits['record_i']]['time']
    assert np.all(result['delta_time'] == true_dt)
