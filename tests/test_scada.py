import numpy as np
import straxen
import warnings


def test_query_sc_values():
    """
    Unity test for the SCADAInterface. Query a fixed range and check if 
    return is correct.
    """

    if not straxen.utilix_is_configured('scada', 'scdata_url'):
        warnings.warn('Cannot test scada since we have no access to xenon secrets.')
        return

    print('Testing SCADAInterface')
    sc = straxen.SCADAInterface(use_progress_bar=False)

    print('Query single value:')
    # Simple query test:
    # Query 5 s of data:
    start = 1609682275000000000
    # Add micro-second to check if query does not fail if inquery precsion > SC precision
    start += 10**6
    end = start + 5 * 10**9
    parameters = {'SomeParameter': 'XE1T.CTPC.Board06.Chan011.VMon'}

    df = sc.get_scada_values(parameters,
                             start=start,
                             end=end,
                             every_nth_value=1,
                             query_type_lab=False,)

    assert df['SomeParameter'][0] // 1 == 1253, 'First values returned is not corrrect.'
    assert np.all(np.isnan(df['SomeParameter'][1:])), 'Subsequent values are not correct.'

    # Test ffill option:
    print('Testing forwardfill option:')
    parameters = {'SomeParameter': 'XE1T.CRY_FCV104FMON.PI'}
    df = sc.get_scada_values(parameters,
                             start=start,
                             end=end,
                             fill_gaps='forwardfill',
                             every_nth_value=1,
                             query_type_lab=False,)
    assert np.all(np.isclose(df[:4], 2.079859)), 'First four values deviate from queried values.'
    assert np.all(np.isclose(df[4:], 2.117820)), 'Last two values deviate from queried values.'

    print('Testing downsampling and averaging option:')
    parameters = {'SomeParameter': 'XE1T.CRY_TE101_TCRYOBOTT_AI.PI'}
    df_all = sc.get_scada_values(parameters,
                                 start=start,
                                 end=end,
                                 fill_gaps='forwardfill',
                                 every_nth_value=1,
                                 query_type_lab=False,)

    df = sc.get_scada_values(parameters,
                             start=start,
                             end=end,
                             fill_gaps='forwardfill',
                             down_sampling=True,
                             every_nth_value=2,
                             query_type_lab=False,)

    assert np.all(df_all[::2] == df), 'Downsampling did not return the correct values.'

    df = sc.get_scada_values(parameters,
                             start=start,
                             end=end,
                             fill_gaps='forwardfill',
                             every_nth_value=2,
                             query_type_lab=False,)

    # Compare average for each second value:
    for ind, i in enumerate([0, 2, 4]):
        assert np.isclose(np.mean(df_all[i:i + 2]), df['SomeParameter'][ind]), 'Averaging is incorrect.'

    # Testing lab query type:
    df = sc.get_scada_values(parameters,
                             start=start,
                             end=end,
                             query_type_lab=True,)

    assert np.all(df['SomeParameter'] // 1 == -96), 'Not all values are correct for query type lab.'
