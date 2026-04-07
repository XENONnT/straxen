"""For all of the context, do a quick check to see that we are able to search a field (i.e. can
build the dependencies in the context correctly) See issue #233 and PR #236."""

import unittest
import strax
import straxen
from straxen.contexts import xenonnt, xenonnt_online, xenonnt_led


##
# XENONnT
##


def test_xenonnt():
    st = xenonnt(_database_init=False)
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
def test_xennonnt_online():
    st = xenonnt_online(_database_init=False)
    st.search_field("time")


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_xenonnt_online_peaklets_chunking_untracked():
    st = xenonnt_online(_database_init=False)
    assert st.config["peaklets_rechunk_on_load"] is False
    assert st.config["peaklets_chunk_target_size_mb"] == strax.DEFAULT_CHUNK_SIZE_MB

    hash_default = st.key_for("0", "peaklets").lineage_hash

    st_rechunk = st.new_context()
    st_rechunk.set_config(
        {
            "peaklets_rechunk_on_load": True,
            "peaklets_chunk_target_size_mb": straxen.Peaklets._chunk_target_size_mb_default,
        }
    )
    hash_rechunk = st_rechunk.key_for("0", "peaklets").lineage_hash

    assert hash_default == hash_rechunk


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_xenonnt_led():
    st = xenonnt_led(_database_init=False)
    st.search_field("time")
