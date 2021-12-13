import straxen
import pandas
import os
import numpy as np
import strax
from matplotlib.pyplot import clf as plt_clf
from straxen.test_utils import nt_test_context, nt_test_run_id
import unittest


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
    # They were added on 25/10/2021 and may be outdated by now
    _expected_test_results = {
        'peak_basics': 40,
        'n_s1': 19,
        'run_live_time': 4.7516763,
        'event_basics': 20,
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.st = nt_test_context()
        # For al the WF plotting, we might need records, let's make those
        cls.st.make(nt_test_run_id, 'records')
        cls.first_peak = cls.st.get_array(nt_test_run_id, 'peak_basics')[0]
        cls.first_event = cls.st.get_array(nt_test_run_id, 'event_basics')[0]

    def tearDown(self):
        plt_clf()

    def test_target_peaks(self, target='peak_basics', tol=2):
        assert target in self._expected_test_results, f'No expectation for {target}?!'
        data = self.st.get_array(nt_test_run_id, target)
        message = (f'Got more/less data for {target}. If you changed something '
                   f'on {target}, please update the numbers in '
                   f'tests/test_mini_analyses.TestMiniAnalyses._expected_test_results')
        if not straxen.utilix_is_configured():
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

    def test_single_event_plot(self):
        plot_all_positions = straxen.utilix_is_configured()
        straxen.analyses.event_display.plot_single_event(
            self.st,
            nt_test_run_id,
            events=self.st.get_array(nt_test_run_id, 'events'),
            event_number=self.first_event['event_number'],
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

    def test_plot_pulses_mv(self):
        self.st.plot_pulses_mv(nt_test_run_id, max_plots=2, plot_hits=True,
                               ignore_time_warning=True)

    def test_plot_pulses_nv(self):
        self.st.plot_pulses_nv(nt_test_run_id, max_plots=2, plot_hits=True,
                               ignore_time_warning=True)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_event_display(self):
        self.st.event_display(nt_test_run_id, time_within=self.first_event)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_event_display_no_rr(self):
        self.st.event_display(nt_test_run_id,
                              time_within=self.first_event,
                              records_matrix=False,
                              event_time_litit=[self.first_event['time'], self.first_event['endtime']],
                              )

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

    @unittest.skipIf(straxen.utilix_is_configured(), "Test for no DB access")
    def test_daq_plot_errors_without_utilix(self):
        with self.assertRaises(NotImplementedError):
            straxen.analyses.daq_waveforms._get_daq_config('som_run', run_collection=None)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_daq_plot_errors(self):
        with self.assertRaises(ValueError):
            straxen.analyses.daq_waveforms._get_daq_config('no_run')
        with self.assertRaises(ValueError):
            straxen.analyses.daq_waveforms._board_to_host_link({'boards': {'boo': 'far'}}, 1)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_event_plot_errors(self):
        with self.assertRaises(ValueError):
            self.st.event_display(records_matrix='records_are_bad')
        with self.assertRaises(ValueError):
            straxen.analyses.event_display._event_display(events=[1, 2, 3],
                                                          context=None,
                                                          to_pe=None,
                                                          run_id=None
                                                          )
        with self.assertRaises(ValueError):
            straxen.analyses.event_display._event_display(axes=None,
                                                          events=[None],
                                                          context=self.st,
                                                          to_pe=None,
                                                          run_id=None,
                                                          )
        with self.assertRaises(strax.DataNotAvailable):
            st_dummy = self.st.new_context()
            st_dummy._plugin_class_regisrty['peaklets'].__version__ = 'lorusipsum'
            straxen.analyses.event_display._event_display(axes=None,
                                                          events=[None],
                                                          context=st_dummy,
                                                          to_pe=None,
                                                          run_id=None,
                                                          )
        with self.assertRaises(ValueError):
            straxen.analyses.event_display.plot_single_event(
                events=[1, 2, 3], event_number=None,)

    def test_interactive_display(self):
        fig = self.st.event_display_interactive(nt_test_run_id,
                                                time_within=self.first_event,
                                                xenon1t=False,
                                                plot_record_matrix=True,
                                                )
        save_as = 'test_display.html'
        fig.save(save_as)
        assert os.path.exists(save_as)
        os.remove(save_as)
        assert not os.path.exists(save_as), f'Should have removed {save_as}'

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
        save_as = 'test_data_selector.html'
        bklt.save(fig, save_as)
        assert os.path.exists(save_as)
        os.remove(save_as)
        assert not os.path.exists(save_as), f'Should have removed {save_as}'

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nt_daq_plot(self):
        self.st.daq_plot(nt_test_run_id,
                         time_within=self.first_peak,
                         vmin=0.1,
                         vmax=1,
                         )

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nt_daq_plot_grouped(self):
        self.st.plot_records_matrix(nt_test_run_id,
                                    time_within=self.first_peak,
                                    vmin=0.1,
                                    vmax=1,
                                    group_by='ADC ID',
                                    )

    def test_records_matrix_downsample(self):
        self.st.records_matrix(nt_test_run_id,
                               time_within=self.first_event,
                               max_samples=20
                               )

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_load_corrected_positions(self):
        self.st.load_corrected_positions(nt_test_run_id, time_within=self.first_peak)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nv_event_display(self):
        self.st.make(nt_test_run_id, 'events_nv')
        self.st.make(nt_test_run_id, 'event_positions_nv')
        with self.assertRaises(ValueError):
            self.st.plot_nveto_event_display(nt_test_run_id, time_within=self.first_peak)


def test_plots():
    """Make some plots"""
    c = np.ones(straxen.n_tpc_pmts)
    straxen.plot_pmts(c)
    straxen.plot_pmts(c, log_scale=True)
