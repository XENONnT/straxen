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
    if not straxen.utilix_is_configured():
        return
    # several fake records do not have any pulse length
    # and channel start at 0, convert to nv:
    records['pulse_length'] = records['length']
    records['channel'] += 2000

    st = straxen.contexts.xenonnt_led()
    plugin = st.get_single_plugin('1', 'ext_timings_nv')
    hits = strax.find_hits(records, min_amplitude=1)

    hitlets = np.zeros(len(hits), strax.hitlet_dtype())
    strax.copy_to_buffer(hits, hitlets, '_refresh_hit_to_hitlets')
    result = plugin.compute(hitlets, records)
    assert len(result) == len(hits)
    assert np.all(result['time'] == hits['time'])
    assert np.all(result['pulse_i'] == hits['record_i'])
    true_dt = hits['time'] - records[hits['record_i']]['time']
    assert np.all(result['delta_time'] == true_dt)
