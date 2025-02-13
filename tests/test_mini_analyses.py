import os
import shutil
import unittest
import platform
import numpy as np
import pandas
import strax
from matplotlib.pyplot import clf
import straxen
from straxen.test_utils import nt_test_context, nt_test_run_id


def is_py310():
    """Check python version."""
    return platform.python_version_tuple()[:2] == ("3", "10")


def test_pmt_pos_nt():
    """Test if we can get the nT PMT positions."""
    pandas.DataFrame(straxen.pmt_positions())


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class TestMiniAnalyses(unittest.TestCase):
    """Generally, tests in this class run st.<some_mini_analysis>

    We provide minimal arguments to just probe if the <some_mini_analysis> is not breaking when
    running, we are NOT checking if plots et cetera make sense, just if the code is not broken (e.g.
    because for changes in dependencies like matplotlib or bokeh)

    NB! If this tests fails locally (but not on github-CI), please do: `rm strax_test_data` You
    might be an old version of test data.

    """

    # They were added on 25/10/2021 and may be outdated by now
    _expected_test_results = {
        "peak_basics": 40,
        "n_s1": 19,
        "run_live_time": 4.7516763,
        "event_basics": 20,
    }

    @classmethod
    def setUpClass(cls) -> None:
        """Common setup for all the tests.

        We need some data which we don't delete but reuse to prevent a lot of computations in this
        class

        """
        cls.st = nt_test_context("xenonnt_online")
        # For al the WF plotting, we might need records, let's make those
        cls.st.make(nt_test_run_id, "records")
        cls.first_peak = cls.st.get_array(nt_test_run_id, "peak_basics")[0]
        cls.first_event = cls.st.get_array(nt_test_run_id, "event_basics")[0]

    @classmethod
    def tearDownClass(cls) -> None:
        """Removes test data after tests are done."""
        path = os.path.abspath(cls.st.storage[-1].path)
        for file in os.listdir(path):
            shutil.rmtree(os.path.join(path, file))

    def tearDown(self):
        """After each test, clear a figure (if one was open)"""
        clf()

    def test_target_peaks(self, target="peak_basics", tol=2):
        """Not a real mini analysis but let's see if the number of peaks matches some pre-defined
        value.

        This is just to safeguard one from accidentally adding some braking code.

        """
        self.assertTrue(target in self._expected_test_results, f"No expectation for {target}?!")
        data = self.st.get_array(nt_test_run_id, target)
        message = (
            f"Got more/less data for {target}. If you changed something "
            f"on {target}, please update the numbers in "
            "tests/test_mini_analyses.TestMiniAnalyses._expected_test_results"
        )
        if not straxen.utilix_is_configured():
            # If we do things with dummy maps, things might be slightly different
            tol += 10
        self.assertTrue(np.abs(len(data) - self._expected_test_results[target]) < tol, message)

    def test_target_events(self):
        """Test that the number of events is roughly right."""
        self.test_target_peaks(target="event_basics")

    def test_plot_waveform(self, deep=False):
        self.st.plot_waveform(nt_test_run_id, time_within=self.first_peak, deep=deep)

    def test_plot_waveform_deep(self):
        self.test_plot_waveform(deep=True)

    def test_plot_hit_pattern(self):
        self.st.plot_hit_pattern(nt_test_run_id, time_within=self.first_peak)

    def test_plot_records_matrix(self):
        self._st_attr_for_one_peak("plot_records_matrix")

    def test_raw_records_matrix(self):
        self._st_attr_for_one_peak("raw_records_matrix")

    def test_event_display_simple(self):
        plot_all_positions = straxen.utilix_is_configured()
        with self.assertRaises(NotImplementedError):
            # old way of calling the simple display
            self.st.event_display_simple(
                nt_test_run_id,
                time_within=self.first_event,
            )
        # New, correct way of calling the simple display
        self.st.event_display(
            nt_test_run_id,
            time_within=self.first_event,
            plot_all_positions=plot_all_positions,
            simple_layout=True,
        )

    def test_single_event_plot(self):
        plot_all_positions = straxen.utilix_is_configured()
        straxen.analyses.event_display.plot_single_event(
            self.st,
            nt_test_run_id,
            events=self.st.get_array(nt_test_run_id, "events"),
            event_number=self.first_event["event_number"],
            plot_all_positions=plot_all_positions,
        )

    def test_event_display_interactive(self):
        self.st.event_display_interactive(
            nt_test_run_id,
            time_within=self.first_event,
        )

    def test_plot_peaks_aft_histogram(self):
        self.st.plot_peaks_aft_histogram(nt_test_run_id)

    def test_event_scatter(self):
        self.st.event_scatter(nt_test_run_id)

    def test_event_scatter_diff_options(self):
        self.st.event_scatter(nt_test_run_id, color_range=(0, 10), color_dim="s1_area")

    def test_energy_spectrum(self):
        self.st.plot_energy_spectrum(nt_test_run_id)

    def test_energy_spectrum_diff_options(self):
        """Run st.plot_energy_spectrum with several options."""
        self.st.plot_energy_spectrum(nt_test_run_id, unit="kg_day_kev", exposure_kg_sec=1)
        self.st.plot_energy_spectrum(nt_test_run_id, unit="tonne_day_kev", exposure_kg_sec=1)
        self.st.plot_energy_spectrum(
            nt_test_run_id, unit="tonne_year_kev", exposure_kg_sec=1, geomspace=False
        )
        with self.assertRaises(ValueError):
            # Some units shouldn't be allowed
            self.st.plot_energy_spectrum(nt_test_run_id, unit="not_allowed_unit", exposure_kg_sec=1)

    def test_peak_classification(self):
        self.st.plot_peak_classification(nt_test_run_id)

    def _st_attr_for_one_peak(self, function_name):
        """Utility function to prevent having to copy past the code below for all the functions we
        are going to test for one peak."""
        f = getattr(self.st, function_name)
        f(nt_test_run_id, time_within=self.first_peak)

    def test_waveform_display(self):
        """Test st.waveform_display for one peak."""
        self._st_attr_for_one_peak("waveform_display")

    def test_hvdisp_plot_pmt_pattern(self):
        """Test st.hvdisp_plot_pmt_pattern for one peak."""
        self._st_attr_for_one_peak("hvdisp_plot_pmt_pattern")

    def test_hvdisp_plot_peak_waveforms(self):
        """Test st.hvdisp_plot_peak_waveforms for one peak."""
        self._st_attr_for_one_peak("hvdisp_plot_peak_waveforms")

    def test_plot_pulses_tpc(self):
        """Test that we can plot some TPC pulses and fail if raise a ValueError if an invalid
        combination of parameters is given."""
        self.st.plot_pulses_tpc(
            nt_test_run_id,
            time_within=self.first_peak,
            max_plots=2,
            plot_hits=True,
            ignore_time_warning=False,
            store_pdf=True,
        )
        with self.assertRaises(ValueError):
            # Raise an error if no time range is specified
            self.st.plot_pulses_tpc(
                nt_test_run_id,
                max_plots=2,
                plot_hits=True,
                ignore_time_warning=True,
                store_pdf=True,
            )

    def test_plot_pulses_mv(self):
        """Repeat above for mv."""
        self.st.plot_pulses_mv(
            nt_test_run_id,
            max_plots=2,
            plot_hits=True,
            ignore_time_warning=True,
        )

    def test_plot_pulses_nv(self):
        """Repeat above for nv."""
        self.st.plot_pulses_nv(
            nt_test_run_id,
            max_plots=2,
            plot_hits=True,
            ignore_time_warning=True,
        )

    def test_event_display(self):
        """Event display plot, needs xedocs."""
        self.st.event_display(nt_test_run_id, time_within=self.first_event)

    def test_event_display_no_rr(self):
        """Make an event display without including records."""
        self.st.event_display(
            nt_test_run_id,
            time_within=self.first_event,
            records_matrix=False,
            event_time_limit=[self.first_event["time"], self.first_event["endtime"]],
        )

    def test_calc_livetime(self):
        """Use straxen.get_livetime_sec."""
        try:
            live_time = straxen.get_livetime_sec(self.st, nt_test_run_id)
        except strax.RunMetadataNotAvailable:
            things = self.st.get_array(nt_test_run_id, "peaks")
            live_time = straxen.get_livetime_sec(self.st, nt_test_run_id, things=things)
        assertion_statement = "Live-time calculation is wrong"
        expected = self._expected_test_results["run_live_time"]
        self.assertTrue(live_time == expected, assertion_statement)

    def test_df_wiki(self):
        """We have a nice utility to write dataframes to the wiki."""
        df = self.st.get_df(nt_test_run_id, "peak_basics")[:10]
        straxen.dataframe_to_wiki(df)

    def test_daq_plot_errors(self):
        """To other ways we should not be allowed to call daq_waveforms.XX."""
        with self.assertRaises(ValueError):
            straxen.analyses.daq_waveforms._get_daq_config("no_run")
        with self.assertRaises(ValueError):
            straxen.analyses.daq_waveforms._board_to_host_link({"boards": [{"no_boards": 0}]}, 1)

    def test_event_plot_errors(self):
        """Several Exceptions should be raised with these following bad ways of calling the event
        display."""
        with self.assertRaises(ValueError):
            # Wrong way of calling records matrix
            self.st.event_display(nt_test_run_id, records_matrix="records_are_bad")
        with self.assertRaises(ValueError):
            # A single event should not have three entries
            straxen.analyses.event_display._event_display(
                events=[1, 2, 3], context=self.st, to_pe=None, run_id="1"
            )
        with self.assertRaises(ValueError):
            # Can't pass empty axes like this to the inner script
            straxen.analyses.event_display._event_display(
                axes=None,
                events=[None],
                context=self.st,
                to_pe=None,
                run_id=nt_test_run_id,
            )
        with self.assertRaises(ValueError):
            # Should raise a valueError
            straxen.analyses.event_display.plot_single_event(
                context=None,
                run_id=None,
                events=[1, 2, 3],
                event_number=None,
            )
        with self.assertRaises(ValueError):
            # Give to many recs to this inner script
            straxen.analyses.event_display._scatter_rec(_event=None, recs=list(range(10)))

    def test_interactive_display(self):
        """Run and save interactive display."""
        fig = self.st.event_display_interactive(
            nt_test_run_id,
            time_within=self.first_event,
            plot_record_matrix=True,
        )
        save_as = "test_display.html"
        fig.save(save_as)
        self.assertTrue(os.path.exists(save_as))
        os.remove(save_as)
        self.assertFalse(os.path.exists(save_as))
        st = self.st.new_context()
        st.event_display_interactive(
            nt_test_run_id,
            time_within=self.first_event,
            plot_record_matrix=False,
            only_main_peaks=True,
        )

    def test_bokeh_selector(self):
        """Test the bokeh data selector."""
        from straxen.analyses.bokeh_waveform_plot import DataSelectionHist

        p = self.st.get_array(nt_test_run_id, "peak_basics", seconds_range=(0, 10))
        with self.assertRaises(NotImplementedError):
            ds = DataSelectionHist("ds")
            fig = ds.histogram2d(
                p,
                p["area"],
                p["area"],
                bins=10,
                hist_range=((0, 200), (0, 2000)),
                log_color_scale=True,
                clim=(10, None),
                undeflow_color="white",
            )

            import bokeh.plotting as bklt

            save_as = "test_data_selector.html"
            bklt.save(fig, save_as)
            self.assertTrue(os.path.exists(save_as))
            os.remove(save_as)
            self.assertFalse(os.path.exists(save_as))
            # Also test if we can write it to the wiki
            straxen.bokeh_to_wiki(fig)
            straxen.bokeh_to_wiki(fig, save_as)
            self.assertTrue(os.path.exists(save_as))
            os.remove(save_as)
            self.assertFalse(os.path.exists(save_as))

    def test_nt_daq_plot(self):
        """Make an nt DAQ plot."""
        self.st.daq_plot(
            nt_test_run_id,
            time_within=self.first_peak,
            vmin=0.1,
            vmax=1,
        )

    def test_nt_daq_plot_grouped(self):
        """Same as above grouped by ADC."""
        self.st.plot_records_matrix(
            nt_test_run_id,
            time_within=self.first_peak,
            vmin=0.1,
            vmax=1,
            group_by="ADC ID",
        )

    def test_records_matrix_downsample(self):
        """Test that downsampling works in the record matrix."""
        self.st.records_matrix(nt_test_run_id, time_within=self.first_event, max_samples=20)

    def test_nv_event_display(self):
        """Test NV event display for a single event."""
        events_nv = self.st.get_array(nt_test_run_id, "events_nv")
        self.st.make(nt_test_run_id, "event_positions_nv")
        self.st.plot_nveto_event_display(
            nt_test_run_id,
            time_within=events_nv[0],
        )
        with self.assertRaises(ValueError):
            # If there is no data, we should raise a ValueError
            self.st.plot_nveto_event_display(
                nt_test_run_id,
                time_range=[-1000, -900],
            )


def test_plots():
    """Make some plots."""
    c = np.ones(straxen.n_tpc_pmts)
    straxen.plot_pmts(c)
    straxen.plot_pmts(c, log_scale=True)
