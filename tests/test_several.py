"""Test several functions distibuted over common.py, misc.py, scada.py"""
import straxen
import pandas
import os
import tempfile
from .test_basics import test_run_id
import numpy as np
import strax
from matplotlib.pyplot import clf as plt_clf

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


# If one of the test below fail, perhaps these values need to be updated.
# They were added on 27/11/2020 and may be outdated by now
EXPECTED_OUTCOMES_TEST_SEVERAL = {
    'n_peaks': 138,
    'n_s1': 4,
    'run_live_time': 0.17933107,
    'n_events': 2
}


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
            st = straxen.contexts.demo()
            print("Get peaks")
            p = st.get_array(test_run_id, 'peaks')

            # Do checks on there number of peaks
            assertion_statement = ("Got /more peaks than expected, perhaps "
                                   "the test is outdated or clustering has "
                                   "really changed")
            assert np.abs(len(p) -
                          EXPECTED_OUTCOMES_TEST_SEVERAL['n_peaks']) < 5, assertion_statement

            st.make(test_run_id, 'event_info')
            print('plot wf')
            peak_i = 0
            st.plot_waveform(test_run_id, time_range=(p[peak_i]['time'], strax.endtime(p[peak_i])))
            plt_clf()

            print('plot hit pattern')
            peak_i = 1
            st.plot_hit_pattern(test_run_id, time_range=(p[peak_i]['time'], strax.endtime(p[peak_i])), xenon1t=True)
            plt_clf()

            print('plot records matrix')
            peak_i = 2
            assert st.is_stored(test_run_id, 'records'), "no records"
            st.plot_records_matrix(test_run_id, time_range=(p[peak_i]['time'], strax.endtime(p[peak_i])))
            plt_clf()

            print('plot aft')
            st.plot_peaks_aft_histogram(test_run_id)
            plt_clf()

            print('plot event scatter')
            st.event_scatter(test_run_id)
            plt_clf()

            print('plot event scatter')
            st.plot_energy_spectrum(test_run_id)
            plt_clf()

            print("Check live-time")
            live_time = straxen.get_livetime_sec(st, test_run_id, things=p)
            assertion_statement = ("Live-time calculation is wrong")
            assert live_time == EXPECTED_OUTCOMES_TEST_SEVERAL['run_live_time'], assertion_statement

            print('Check the peak_basics')
            df = st.get_df(test_run_id, 'peak_basics')
            assertion_statement = ("Got less/more S1s than expected, perhaps "
                                   "the test is outdated or classification "
                                   "has really changed.")
            assert np.abs(np.sum(df['type'] == 1) -
                          EXPECTED_OUTCOMES_TEST_SEVERAL['n_s1']) < 2, assertion_statement
            df = df[:10]

            print("Check that we can write nice wiki dfs")
            straxen.dataframe_to_wiki(df)

            print("Abbuse the peaks to show that _average_scada works")
            p = p[:10]
            p_t, p_a = straxen.scada._average_scada(
                p['time']/1e9,
                p['time'],
                1)
            assert len(p_a) == len(p), 'Scada deleted some of my 10 peaks!'

            print('Check the number of events')
            events = st.get_array(test_run_id, 'event_info')
            assertion_statement = ("Got less/ore events than expected, "
                                   "perhaps the test is outdated or something "
                                   "changed in the processing.")
            assert len(events) == EXPECTED_OUTCOMES_TEST_SEVERAL['n_events'], assertion_statement
        # On windows, you cannot delete the current process'
        # working directory, so we have to chdir out first.
        finally:
            os.chdir('..')


def test_plots():
    """Make some plots"""
    c = np.ones(straxen.n_tpc_pmts)
    straxen.plot_pmts(c)
    straxen.plot_pmts(c, log_scale=True)
