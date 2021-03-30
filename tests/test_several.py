"""Test several functions distibuted over common.py, misc.py, scada.py"""
import straxen
import pandas
import os
import tempfile
from .test_basics import test_run_id_1T
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
            st.make(test_run_id_1T, 'records')
            # Ignore strax-internal warnings
            st.set_context_config({'free_options': tuple(st.config.keys())})
            st.make(test_run_id_1T, 'records')

            print("Get peaks")
            p = st.get_array(test_run_id_1T, 'peaks')

            # Do checks on there number of peaks
            assertion_statement = ("Got /more peaks than expected, perhaps "
                                   "the test is outdated or clustering has "
                                   "really changed")
            assert np.abs(len(p) -
                          EXPECTED_OUTCOMES_TEST_SEVERAL['n_peaks']) < 5, assertion_statement

            events = st.get_array(test_run_id_1T, 'event_info')
            print('plot wf')
            peak_i = 0
            st.plot_waveform(test_run_id_1T, time_range=(p[peak_i]['time'], strax.endtime(p[peak_i])))
            plt_clf()

            print('plot hit pattern')
            peak_i = 1
            st.plot_hit_pattern(test_run_id_1T, time_range=(p[peak_i]['time'], strax.endtime(p[peak_i])), xenon1t=True)
            plt_clf()

            print('plot (raw)records matrix')
            peak_i = 2
            assert st.is_stored(test_run_id_1T, 'records'), "no records"
            assert st.is_stored(test_run_id_1T, 'raw_records'), "no raw records"
            st.plot_records_matrix(test_run_id_1T, time_range=(p[peak_i]['time'],
                                                               strax.endtime(p[peak_i])))

            st.raw_records_matrix(test_run_id_1T, time_range=(p[peak_i]['time'],
                                                           strax.endtime(p[peak_i])))
            st.plot_waveform(test_run_id_1T,
                             time_range=(p[peak_i]['time'],
                                         strax.endtime(p[peak_i])),
                             deep=True)
            plt_clf()

            print('plot event displays')
            straxen.analyses.event_display.plot_single_event(st,
                                                             test_run_id_1T,
                                                             events,
                                                             xenon1t=True,
                                                             event_number=0,
                                                             records_matrix=True)
            plt_clf()
            straxen.analyses.event_display.plot_single_event(st,
                                                             test_run_id_1T,
                                                             events,
                                                             xenon1t=True,
                                                             event_number=0,
                                                             records_matrix='raw')
            plt_clf()
            st.event_display_interactive(test_run_id_1T, time_range=(events[0]['time'],
                                                                  events[0]['endtime']),
                                         xenon1t=True)
            plt_clf()

            print('plot aft')
            st.plot_peaks_aft_histogram(test_run_id_1T)
            plt_clf()

            print('plot event scatter')
            st.event_scatter(test_run_id_1T)
            plt_clf()

            print('plot event scatter')
            st.plot_energy_spectrum(test_run_id_1T)
            plt_clf()

            print('plot peak clsassification')
            st.plot_peak_classification(test_run_id_1T)
            plt_clf()

            print("plot holoviews")
            peak_i = 3
            st.waveform_display(test_run_id_1T,
                                time_range=(p[peak_i]['time'],
                                            strax.endtime(p[peak_i])))
            st.hvdisp_plot_pmt_pattern(test_run_id_1T,
                                time_range=(p[peak_i]['time'],
                                            strax.endtime(p[peak_i])))
            st.hvdisp_plot_peak_waveforms(test_run_id_1T,
                                time_range=(p[peak_i]['time'],
                                            strax.endtime(p[peak_i])))


            print('Plot single pulse:')
            st.plot_pulses_tpc(test_run_id_1T, max_plots=2,  plot_hits=True, ignore_time_warning=True)

            print("Check live-time")
            live_time = straxen.get_livetime_sec(st, test_run_id_1T, things=p)
            assertion_statement = "Live-time calculation is wrong"
            assert live_time == EXPECTED_OUTCOMES_TEST_SEVERAL['run_live_time'], assertion_statement

            print('Check the peak_basics')
            df = st.get_df(test_run_id_1T, 'peak_basics')
            assertion_statement = ("Got less/more S1s than expected, perhaps "
                                   "the test is outdated or classification "
                                   "has really changed.")
            assert np.abs(np.sum(df['type'] == 1) -
                          EXPECTED_OUTCOMES_TEST_SEVERAL['n_s1']) < 2, assertion_statement
            df = df[:10]

            print("Check that we can write nice wiki dfs")
            straxen.dataframe_to_wiki(df)

            print("Abuse the peaks to show that _average_scada works")
            p = p[:10]
            p_t, p_a = straxen.scada._average_scada(
                p['time']/1e9,
                p['time'],
                1)
            assert len(p_a) == len(p), 'Scada deleted some of my 10 peaks!'

            print('Check the number of events')
            events = st.get_array(test_run_id_1T, 'event_info_double')
            assertion_statement = ("Got less/ore events than expected, "
                                   "perhaps the test is outdated or something "
                                   "changed in the processing.")
            assert len(events) == EXPECTED_OUTCOMES_TEST_SEVERAL['n_events'], assertion_statement

            print("Plot bokkeh:")
            fig = st.event_display_interactive(test_run_id_1T,
                                               time_range=(events[0]['time'],
                                                           events[0]['endtime']),
                                               xenon1t=True,
                                               plot_record_matrix=True,
                                               )
            fig.save('test_display.html')

            # Test data selector:
            from straxen.analyses.bokeh_waveform_plot import DataSelectionHist
            ds = DataSelectionHist('ds')
            fig = ds.histogram2d(p,
                               p['area'],
                               p['area'],
                               bins=50,
                               hist_range=((0, 200), (0, 2000)),
                               log_color_scale=True,
                               clim=(10, None),
                               undeflow_color='white')

            import bokeh.plotting as bklt
            bklt.save(fig, 'test_data_selector.html')

        # On windows, you cannot delete the current process'
        # working directory, so we have to chdir out first.
        finally:
            os.chdir('..')


def test_plots():
    """Make some plots"""
    c = np.ones(straxen.n_tpc_pmts)
    straxen.plot_pmts(c)
    straxen.plot_pmts(c, log_scale=True)


def test_print_version():
    straxen.print_versions(['strax', 'something_that_does_not_exist'])


def test_nt_minianalyses():
    """Number of tests to be run on nT like configs"""
    if not straxen.utilix_is_configured():
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("Temporary directory is ", temp_dir)
            os.chdir(temp_dir)
            from .test_plugins import DummyRawRecords, testing_config_nT, test_run_id_nT
            st = straxen.contexts.xenonnt_online()
            rundb = st.storage[0]
            rundb.readonly = True
            st.storage = [rundb, strax.DataDirectory(temp_dir)]

            # We want to test the FDC map that only works with CMT
            test_conf = testing_config_nT.copy()
            del test_conf['fdc_map']

            st.set_config(test_conf)
            st.set_context_config(dict(forbid_creation_of=()))
            st.register(DummyRawRecords)

            rr = st.get_array(test_run_id_nT, 'raw_records')
            st.make(test_run_id_nT, 'records')
            st.make(test_run_id_nT, 'peak_basics')

            st.daq_plot(test_run_id_nT,
                        time_range=(rr['time'][0], strax.endtime(rr)[-1]),
                        vmin=0.1,
                        vmax=1,
                        )

            st.plot_records_matrix(test_run_id_nT,
                                   time_range=(rr['time'][0],
                                               strax.endtime(rr)[-1]),
                                   vmin=0.1,
                                   vmax=1,
                                   group_by='ADC ID',
                                   )
            plt_clf()

            st.make(test_run_id_nT, 'event_info')
            st.load_corrected_positions(test_run_id_nT,
                                        time_range=(rr['time'][0],
                                                    strax.endtime(rr)[-1]),

                                        )
            # This would be nice to add but with empty events it does not work
            # st.event_display(test_run_id_nT,
            #                  time_range=(rr['time'][0],
            #                              strax.endtime(rr)[-1]),
            #                  )
        # On windows, you cannot delete the current process'git p
        # working directory, so we have to chdir out first.
        finally:
            os.chdir('..')
