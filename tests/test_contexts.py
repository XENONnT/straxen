"""For all of the context, do a quick check to see that we are able to search
a field (i.e. can build the dependencies in the context correctly)
See issue #233 and PR #236"""
from straxen.contexts import xenon1t_dali, xenon1t_led, fake_daq, demo
from straxen.contexts import xenonnt_led, xenonnt_online, xenonnt
import straxen
import tempfile
import os


##
# XENONnT
##


def test_xenonnt_online():
    st = xenonnt_online(_database_init=False, use_rucio=False)
    st.search_field('time')


def test_xennonnt():
    if straxen.utilix_is_configured():
        st = xenonnt(_database_init=False, use_rucio=False)
        st.search_field('time')


def test_xenonnt_led():
    st = xenonnt_led(_database_init=False, use_rucio=False)
    st.search_field('time')


def test_nt_is_nt_online():
    if not straxen.utilix_is_configured():
        # Cannot contact CMT without the database
        return
    # Test that nT and nT online are the same
    st_online = xenonnt_online(_database_init=False, use_rucio=False)

    st = xenonnt(_database_init=False, use_rucio=False)
    for plugin in st._plugin_class_registry.keys():
        print(f'Checking {plugin}')
        nt_key = st.key_for('0', plugin)
        nt_online_key = st_online.key_for('0', plugin)
        assert str(nt_key) == str(nt_online_key)


##
# XENON1T
##


def test_xenon1t_dali():
    st = xenon1t_dali()
    st.search_field('time')


def test_demo():
    """
    Test the demo context. Since we download the folder to the current
        working directory, make sure we are in a tempfolder where we
        can write the data to
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("Temporary directory is ", temp_dir)
            os.chdir(temp_dir)
            st = demo()
            st.search_field('time')
        # On windows, you cannot delete the current process'
        # working directory, so we have to chdir out first.
        finally:
            os.chdir('..')


def test_fake_daq():
    st = fake_daq()
    st.search_field('time')


def test_xenon1t_led():
    st = xenon1t_led()
    st.search_field('time')
