"""Testing functions for the CMT services"""

import straxen
import numpy as np


def test_mc_wrapper_elife(run_id='009000',
                          cmt_id='016000',
                          mc_id='mc_0',
                          ):
    """
    Test that for two different run ids, we get different elifes using
    the MC wrapper.
    :param run_id: First run-id (used for normal query)
    :param cmt_id: Second run-id used as a CMT id (should not be the
        same as run_id! otherwise the values might actually be the same
        and the test does not work).
    :return: None
    """
    if not straxen.utilix_is_configured():
        return
    assert np.abs(int(run_id) - int(cmt_id)) > 500, 'runs must be far apart'

    # First for the run-id let's get the value
    elife = straxen.get_correction_from_cmt(
        run_id,
        ("elife", "ONLINE", True))

    # Now, we repeat the same query using the MC wrapper, this should
    # give us a different result since we are now asking for a very
    # different run-number.
    mc_elife_diff = straxen.get_correction_from_cmt(
        mc_id,
        ('MC', cmt_id, "elife", "ONLINE", True)
    )

    # Repeat the query from above to verify, let's see if we are getting
    # the same results as for `elife` above
    mc_elife_same = straxen.get_correction_from_cmt(
        mc_id,
        ('MC', run_id, "elife", "ONLINE", True)
    )

    assert elife != mc_elife_diff
    assert elife == mc_elife_same


def test_mc_wrapper_gains(run_id='009000',
                          cmt_id='016000',
                          mc_id='mc_0',
                          execute=True,
                          ):
    """
    Test that for two different run ids, we get different gains using
    the MC wrapper.
    :param run_id: First run-id (used for normal query)
    :param cmt_id: Second run-id used as a CMT id (should not be the
        same as run_id! otherwise the values might actually be the same
        and the test does not work).
    :param execute: Execute this test (this is set to False since the
        test takes 9 minutes which is too long. We can activate this if
        the testing time due to faster CMT queries is reduced).
    :return: None
    """
    if not straxen.utilix_is_configured() or not execute:
        return

    assert np.abs(int(run_id) - int(cmt_id)) > 500, 'runs must be far apart'

    # First for the run-id let's get the value
    gains = straxen.get_to_pe(
        run_id,
        ('CMT_model', ('to_pe_model', 'ONLINE')),
        straxen.n_tpc_pmts)

    # Now, we repeat the same query using the MC wrapper, this should
    # give us a different result since we are now asking for a very
    # different run-number.
    mc_gains_diff = straxen.get_to_pe(
        mc_id,
        ('MC', cmt_id, 'CMT_model', ('to_pe_model', 'ONLINE')),
        straxen.n_tpc_pmts)

    # Repeat the query from above to verify, let's see if we are getting
    # the same results as for `gains` above
    mc_gains_same = straxen.get_to_pe(
        mc_id,
        ('MC', run_id, 'CMT_model', ('to_pe_model', 'ONLINE')),
        straxen.n_tpc_pmts)

    assert not np.all(gains == mc_gains_diff)
    assert np.all(gains == mc_gains_same)
