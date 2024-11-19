"""For all of the context, do a quick check to see that we are able to search a field (i.e. can
build the dependencies in the context correctly) See issue #233 and PR #236."""

import unittest
import straxen
from straxen.contexts import xenonnt_led, xenonnt_online, xenonnt


##
# XENONnT
##


def test_xenonnt_online():
    st = xenonnt_online(_database_init=False)
    st.search_field("time")


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_xenonnt_online_with_online_frontend():
    st = xenonnt_online(include_online_monitor=True)
    for sf in st.storage:
        if "OnlineMonitor" == sf.__class__.__name__:
            break
    else:
        raise ValueError(f"Online monitor not in {st.storage}")


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_xenonnt_online_rucio_local():
    st = xenonnt_online(include_rucio_local=True, _rucio_local_path="./test")
    for sf in st.storage:
        if "RucioLocalFrontend" == sf.__class__.__name__:
            break
    else:
        raise ValueError(f"Online monitor not in {st.storage}")


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_xennonnt():
    st = xenonnt(_database_init=False)
    st.search_field("time")


def test_xenonnt_led():
    st = xenonnt_led(_database_init=False)
    st.search_field("time")


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_nt_is_nt_online():
    # Test that nT and nT online are the same
    st_online = xenonnt_online(_database_init=False)

    st = xenonnt(_database_init=False)
    for plugin in st._plugin_class_registry.keys():
        print(f"Checking {plugin}")
        nt_key = st.key_for("0", plugin)
        nt_online_key = st_online.key_for("0", plugin)
        assert str(nt_key) == str(nt_online_key)


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_cmt_versions():
    """Let's try and see which CMT versions are compatible with this straxen version."""
    cmt = straxen.CorrectionsManagementServices()
    cmt_versions = list(cmt.global_versions)[::-1]
    print(cmt_versions)
    success_for = []
    for global_version in cmt_versions:
        try:
            xenonnt(global_version)
            success_for.append(global_version)
        except straxen.CMTVersionError:
            pass
    print(
        f"This straxen version works with {success_for} but is "
        f"incompatible with {set(cmt_versions) - set(success_for)}"
    )

    test = unittest.TestCase()
    # We should always work for one offline and the online version
    test.assertTrue(len(success_for) >= 2)
