"""For all of the context, do a quick check to see that we are able to search
a field (i.e. can build the dependencies in the context correctly)
See issue #233 and PR #236"""
from straxen.contexts import *


def import_wfsim():
    """Check if we can test the wfsim-related contexts. This is not the case
    for e.g. travis checks as we don't install wfsim by default."""
    try:
        import wfsim
        return True
    except ImportError:
        # We cannot test the wfsim as it is not installed.
        return False

##
# XENONnT
##


def test_xenonnt_online():
    st = xenonnt_online(_database_init=False)
    st.search_field('time')


def test_xenonnt_led():
    st = xenonnt_led(_database_init=False)
    st.search_field('time')


def xenon_xenonnt_simulation():
    if import_wfsim():
        st = xenonnt_simulation()
        st.search_field('time')


def xenonnt_temporary_five_pmts():
    st = xenonnt_temporary_five_pmts(_database_init=False)
    st.search_field('time')


##
# XENON1T
##

def test_xenon1t_dali():
    st = xenon1t_dali()
    st.search_field('time')


def test_demo():
    st = demo()
    st.search_field('time')


def test_fake_daq():
    st = fake_daq()
    st.search_field('time')


def test_xenon1t_led():
    st = xenon1t_led()
    st.search_field('time')


def xenon_xenon1t_simulation():
    if import_wfsim():
        st = xenon1t_simulation()
        st.search_field('time')
