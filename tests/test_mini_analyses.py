"""Test several functions distibuted over common.py, misc.py, scada.py"""
import straxen
import pandas
import os
import tempfile
import numpy as np
import strax
from matplotlib.pyplot import clf as plt_clf
from straxen.test_utils import nt_test_context, nt_test_run_id
import unittest

# If one of the test below fail, perhaps these values need to be updated.


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


class TestMiniAnalyses(unittest.TestCase):
    """

    """
    # They were added on 27/11/2020 and may be outdated by now
    _expected_test_results = {
        'peak_basics': 40,
        'n_s1': 19,
        'run_live_time': 4.7516763,
        'event_basics': 20,
    }

    def setUp(self) -> None:
        self.st = nt_test_context()
        # For al the WF plotting, we might need records, let's make those
        self.st.make(nt_test_run_id, 'records')
        if not hasattr(self, 'first_event') or not hasattr(self, 'event_basics'):
            self.first_peak = self.st.get_array(nt_test_run_id, 'peak_basics')[0]
            self.first_event = self.st.get_array(nt_test_run_id, 'event_basics')[0]

    def tearDown(self):
        plt_clf()

    def test_target_peaks(self, target='peak_basics', tol=2):
        assert target in self._expected_test_results, f'No expectation for {target}?!'
        data = self.st.get_array(nt_test_run_id, target)
        message = (f'Got more/less data for {target}. If you changed something '
                   f'on {target}, please update the numbers in '
                   f'tests/test_mini_analyses.TestMiniAnalyses._expected_test_results')
        if not straxen.utilix_is_configured(warning_message=None):
            # If we do things with dummy maps, things might be slightly different
           tol += 10
        assert np.abs(len(data) - self._expected_test_results[target]) < tol, message

    def test_target_events(self):
        self.test_target_peaks(target='event_basics')

    def test_plot_waveform(self, deep=False):
        self.st.plot_waveform(nt_test_run_id, time_within=self.first_peak, deep=deep)

    def test_plot_waveform_deep(self):
        self.test_plot_waveform(deep=True)

    def test_plot_hit_pattern(self):
        self.st.plot_hit_pattern(nt_test_run_id, time_within=self.first_peak, xenon1t=False)

    def test_plot_records_matrix(self):
        self.st_attr_for_one_peak('plot_records_matrix')

    def test_raw_records_matrix(self):
        self.st_attr_for_one_peak('raw_records_matrix')

    def test_event_display_simple(self):
        plot_all_positions = straxen.utilix_is_configured()
        self.st.event_display_simple(nt_test_run_id,
                                time_within=self.first_event,
                                xenon1t=False,
                                plot_all_positions=plot_all_positions,
                                )

    def test_event_display_interactive(self):
        self.st.event_display_interactive(nt_test_run_id,
                                time_within=self.first_event,
                                xenon1t=False,
                                )

    def test_plot_peaks_aft_histogram(self):
        self.st.plot_peaks_aft_histogram(nt_test_run_id)

    def test_event_scatter(self):
        self.st.event_scatter(nt_test_run_id)

    def test_energy_spectrum(self):
        self.st.plot_energy_spectrum(nt_test_run_id)

    def test_peak_classification(self):
        self.st.plot_peak_classification(nt_test_run_id)

    def st_attr_for_one_peak(self, function_name):
        f = getattr(self.st, function_name)
        f(nt_test_run_id, time_within=self.first_peak)

    def test_waveform_display(self):
        self.st_attr_for_one_peak('waveform_display')

    def test_hvdisp_plot_pmt_pattern(self):
        self.st_attr_for_one_peak('hvdisp_plot_pmt_pattern')

    def test_hvdisp_plot_peak_waveforms(self):
        self.st_attr_for_one_peak('hvdisp_plot_peak_waveforms')

    def test_plot_pulses_tpc(self):
        self.st.plot_pulses_tpc(nt_test_run_id, max_plots=2, plot_hits=True,
                           ignore_time_warning=True)

    def test_calc_livetime(self):
        try:
            live_time = straxen.get_livetime_sec(self.st, nt_test_run_id)
        except strax.RunMetadataNotAvailable:
            things = self.st.get_array(nt_test_run_id, 'peaks')
            live_time = straxen.get_livetime_sec(self.st, nt_test_run_id, things=things)
        assertion_statement = "Live-time calculation is wrong"
        assert live_time == self._expected_test_results['run_live_time'], assertion_statement

    def test_df_wiki(self):
        df = self.st.get_df(nt_test_run_id, 'peak_basics')
        straxen.dataframe_to_wiki(df)

    def test_interactive_display(self):

        fig = self.st.event_display_interactive(nt_test_run_id,
                                           time_within=self.first_event,
                                           xenon1t=False,
                                           plot_record_matrix=True,
                                           )
        fig.save('test_display.html')

    def test_selector(self):
        from straxen.analyses.bokeh_waveform_plot import DataSelectionHist
        p = self.st.get_array(nt_test_run_id, 'peak_basics')
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

def test_plots():
    """Make some plots"""
    c = np.ones(straxen.n_tpc_pmts)
    straxen.plot_pmts(c)
    straxen.plot_pmts(c, log_scale=True)


def test_print_version():
    straxen.print_versions(['strax', 'something_that_does_not_exist'])


# def test_nd_daq_plot():
#     """Number of tests to be run on nT like configs"""
#     if not straxen.utilix_is_configured():
#         return
#     with tempfile.TemporaryDirectory() as temp_dir:
#         try:
#             print("Temporary directory is ", temp_dir)
#             os.chdir(temp_dir)
#             st = straxen.contexts.xenonnt_online(use_rucio=False)
#             rundb = st.storage[0]
#             rundb.readonly = True
#             st.storage = [rundb, strax.DataDirectory(temp_dir)]
#
#             # We want to test the FDC map that only works with CMT
#             test_conf = straxen.test_utils._testing_config_nT.copy()
#             del test_conf['fdc_map']
#
#             st.set_config(test_conf)
#             st.set_context_config(dict(forbid_creation_of=()))
#             st.register(DummyRawRecords)
#
#             rr = st.get_array(nt_test_run_id, 'raw_records')
#             st.make(nt_test_run_id, 'records')
#             st.make(nt_test_run_id, 'peak_basics')
#
#             plt_clf()
#         # On windows, you cannot delete the current process'git p
#         # working directory, so we have to chdir out first.
#         finally:
#             os.chdir('..')


def test_nd_daq_plot():
    """Number of tests to be run on nT like configs"""
    if not straxen.utilix_is_configured():
        return
    st = straxen.test_utils.nt_test_context(use_rucio=False)
    ev = st.get_array(nt_test_run_id, 'event_info')
    st.load_corrected_positions(nt_test_run_id,
                                time_within=ev[0],
                                )
    # This would be nice to add but with empty events it does not work
    if len(ev):
        st.event_display(nt_test_run_id,
                         time_within=ev[0],
                         )
    rr = st.get_array(nt_test_run_id, 'raw_records')
    st.make(nt_test_run_id, 'records')
    st.make(nt_test_run_id, 'peak_basics')
    st.daq_plot(nt_test_run_id,
                time_range=(rr['time'][0], strax.endtime(rr)[-1]),
                vmin=0.1,
                vmax=1,
                )

    st.plot_records_matrix(nt_test_run_id,
                           time_range=(rr['time'][0],
                                       strax.endtime(rr)[-1]),
                           vmin=0.1,
                           vmax=1,
                           group_by='ADC ID',
                           )
