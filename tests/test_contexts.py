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
