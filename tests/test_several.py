"""Test several functions distibuted over common.py, misc.py, scada.py"""
import straxen
import pandas
import os
import tempfile
from .test_basics import test_run_id
import numpy as np


def test_pmt_pos_1t():
    """
    Test if we can get the 1T PMT positions
    """
    pandas.DataFrame(straxen.pmt_positions(True))


def test_pmt_pos_nt():
    """
    Test if we can get the nT PMT positions
    """
    pandas.DataFrame(straxen.pmt_positions(False))


def test_secret():
    """
    Check something in the sectets. This should not work because we
    don't have any.
    """
    try:
        straxen.get_secret('somethingnonexistent')
    except ValueError:
        # Good we got some message we cannot load something that does
        # not exist,
        pass


def test_several():
    """
    Test several other functions in straxen. Is kind of messy but saves
    time as we won't load data many times
    :return:
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("Temporary directory is ", temp_dir)
            os.chdir(temp_dir)
            print("Downloading test data (if needed)")

            print("Get peaks")
            st = straxen.contexts.demo()
            p = st.get_array(test_run_id, 'peaks')
            print("Check live-time")
            straxen.get_livetime_sec(st, test_run_id, things=p)

            print("Check that we can write nice wiki dfs")
            df = st.get_df(test_run_id, 'peak_basics')
            df = df[:10]
            straxen.dataframe_to_wiki(df)

            print("Abbuse the peaks to show that _average_scada works")
            p = p[:10]
            p_t, p_a = straxen.scada._average_scada(
                p['time']/1e9,
                p['time'],
                1)

            assert len(p_a) == len(p), 'Scada deleted some of my 10 peaks!'

        # On windows, you cannot delete the current process'
        # working directory, so we have to chdir out first.
        finally:
            os.chdir('..')


def test_plots():
    """Make some plots"""
    c = np.ones(straxen.n_tpc_pmts)
    straxen.plot_pmts(c)
    straxen.plot_pmts(c, log_scale=True)
