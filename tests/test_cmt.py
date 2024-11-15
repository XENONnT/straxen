"""Testing functions for the CMT services."""

import strax
import straxen
import utilix
import numpy as np
from straxen.test_utils import nt_test_run_id as test_run_id_nT
import unittest


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_connect_to_db():
    """Test connection to db."""
    corrections_collection = utilix.rundb.xent_collection(database="corrections")
    client = corrections_collection.database.client
    cmt = strax.CorrectionsInterface(client, database_name="corrections")
    df = cmt.read("global_xenonnt")
    mes = "Return empty dataframe when reading DB. Please check"
    assert not df.empty, mes


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_cmt_conf_option(option="mlp_model", version="ONLINE", is_nT=True):
    """Test CMT conf options If wrong conf is passed it would raise an error accordingly."""
    conf = option, version, is_nT
    correction = straxen.get_correction_from_cmt(test_run_id_nT, conf)
    assert isinstance(correction, (float, int, str, np.ndarray))


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_mc_wrapper_elife(
    run_id="009000",
    cmt_id="016000",
    mc_id="mc_0",
):
    """Test that for two different run ids, we get different elifes using the MC wrapper.

    :param run_id: First run-id (used for normal query)
    :param cmt_id: Second run-id used as a CMT id (should not be the same as run_id! otherwise the
        values might actually be the same and the test does not work).
    :return: None

    """
    assert np.abs(int(run_id) - int(cmt_id)) > 500, "runs must be far apart"

    # First for the run-id let's get the value
    elife = straxen.get_correction_from_cmt(run_id, ("elife", "ONLINE", True))

    # Now, we repeat the same query using the MC wrapper, this should
    # give us a different result since we are now asking for a very
    # different run-number.
    mc_elife_diff = straxen.get_correction_from_cmt(
        mc_id, ("cmt_run_id", cmt_id, "elife", "ONLINE", True)
    )

    # Repeat the query from above to verify, let's see if we are getting
    # the same results as for `elife` above
    mc_elife_same = straxen.get_correction_from_cmt(
        mc_id, ("cmt_run_id", run_id, "elife", "ONLINE", True)
    )

    assert elife != mc_elife_diff
    assert elife == mc_elife_same


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_mc_wrapper_gains(
    run_id="009000",
    cmt_id="016000",
    mc_id="mc_0",
    execute=True,
):
    """Test that for two different run ids, we get different gains using the MC wrapper.

    :param run_id: First run-id (used for normal query)
    :param cmt_id: Second run-id used as a CMT id (should not be the same as run_id! otherwise the
        values might actually be the same and the test does not work).
    :param execute: Execute this test (this is set to False since the test takes 9 minutes which is
        too long. We can activate this if the testing time due to faster CMT queries is reduced).
    :return: None

    """
    if not execute:
        return
    assert np.abs(int(run_id) - int(cmt_id)) > 500, "runs must be far apart"

    # First for the run-id let's get the value
    gains = straxen.get_correction_from_cmt(run_id, ("to_pe_model", "ONLINE", True))

    # Now, we repeat the same query using the MC wrapper, this should
    # give us a different result since we are now asking for a very
    # different run-number.
    mc_gains_diff = straxen.get_correction_from_cmt(
        mc_id, ("cmt_run_id", cmt_id, "to_pe_model", "ONLINE", True)
    )

    # Repeat the query from above to verify, let's see if we are getting
    # the same results as for `gains` above
    mc_gains_same = straxen.get_correction_from_cmt(
        mc_id, ("cmt_run_id", run_id, "to_pe_model", "ONLINE", True)
    )

    assert not np.all(gains == mc_gains_diff)
    assert np.all(gains == mc_gains_same)


def test_is_cmt_option():
    """Catches if we change the CMT option structure.

    The example dummy_option works at least before Jun 13 2021

    """
    dummy_option = ("hit_thresholds_tpc", "ONLINE", True)
    assert straxen.is_cmt_option(dummy_option), "Structure of CMT options changed!"

    dummy_url_config = "cmt://correction?version=ONLINE&run_id=plugin.run_id"
    assert straxen.is_cmt_option(dummy_url_config), "Structure of CMT options changed!"


def test_replace_url_version():
    """Tests the replace_url_version function which is important in apply_cmt_version."""
    url = "cmt://elife?version=ONLINE&run_id=plugin.run_id"
    url_check = "cmt://elife?version=v1&run_id=plugin.run_id"
    url_test = straxen.corrections_services.replace_url_version(url, "v1")
    if url_check != url_test:
        raise AssertionError(
            "replace_url_version did not do its job! "
            f"it returns:\n{url_test}\nwhen it should return:\n{url_check}"
        )
