import strax
import straxen
from straxen.holoviews_utils import nVETOEventDisplay
from straxen.holoviews.holoviews_peak_data import PlotPeaksTPC
from straxen.holoviews.holoviews_pmt_array import PlotPMTArrayTPC
from straxen.holoviews.holoviews_inspector import EventStats
import holoviews as hv
import panel as pn
import numpy as np
from tempfile import TemporaryDirectory
import os
import pandas as pd


_nveto_pmt_dummy = {'channel': list(range(2000, 2120)),
                    'x': list(range(120)),
                    'y': list(range(120)),
                    'z': list(range(120)),
                    }
_dummy_map = pd.DataFrame(_nveto_pmt_dummy).to_records()


def test_hitlets_to_hv_points():
    hit = np.zeros(1, dtype=strax.hit_dtype)
    hit['time'] = 10
    hit['length'] = 2
    hit['dt'] = 1
    hit['channel'] = 2000
    hit['area'] = 1

    nvd = nVETOEventDisplay(pmt_map=_dummy_map)
    points = nvd.hitlets_to_hv_points(hit, t_ref=0)

    m = [hit[key] == points.data[key] for key in hit.dtype.names if
         key in points.data.columns.values]
    assert np.all(m), 'Data has not been converted corretly into hv.Points.'


def test_hitlet_matrix():
    hit = np.zeros(1, dtype=strax.hit_dtype)
    hit['time'] = 10
    hit['length'] = 2
    hit['dt'] = 1
    hit['channel'] = 2000
    hit['area'] = 1

    nvd = nVETOEventDisplay(pmt_map=_dummy_map)
    hit_m = nvd.plot_hitlet_matrix(hitlets=hit)

    with TemporaryDirectory() as d:
        # Have to store plot to make sure it is rendered
        hv.save(hit_m, os.path.join(d, 'hitlet_matrix.html'))


def test_plot_nveto_pattern():
    hit = np.zeros(1, dtype=strax.hit_dtype)
    hit['channel'] = 2000
    hit['area'] = 1

    nvd = nVETOEventDisplay(pmt_map=_dummy_map)
    pmt_plot = nvd.plot_nveto(hitlets=hit)
    with TemporaryDirectory() as d:
        # Have to store plot to make sure it is rendered
        hv.save(pmt_plot, os.path.join(d, 'hitlet_matrix.html'))


def test_nveto_event_display():
    hit = np.zeros(1, dtype=strax.hit_dtype)
    hit['time'] = 10
    hit['length'] = 2
    hit['dt'] = 1
    hit['channel'] = 2000
    hit['area'] = 1

    dtype = straxen.plugins.events_nv.veto_event_dtype()
    dtype += straxen.plugins.event_positions_nv.veto_event_positions_dtype()[2:]
    event = np.zeros(1, dtype=dtype)
    event['time'] = hit['time']
    event['endtime'] = hit['time'] + 40
    event['area'] = hit['area']

    nvd = nVETOEventDisplay(event, hit, pmt_map=_dummy_map, run_id='014986')
    dispaly = nvd.plot_event_display()

    with TemporaryDirectory() as d:
        # Have to store plot to make sure it is rendered
        pn.io.save.save(dispaly, os.path.join(d, 'hitlet_matrix.html'))


def test_array_to_df_and_make_sliders():
    dtype = (straxen.plugins.events_nv.veto_event_dtype()
             + straxen.plugins.event_positions_nv.veto_event_positions_dtype()[2:])
    evt = np.zeros(1, dtype)

    nvd = nVETOEventDisplay(pmt_map=_dummy_map)
    df = straxen.convert_array_to_df(evt)

    nvd._make_sliders_and_tables(df)


def test_static_detector_plots():
    tpc = straxen.holoviews_utils.plot_tpc_circle(straxen.cryostat_outer_radius)
    diffuer_balls = straxen.holoviews_utils.plot_diffuser_balls_nv()
    nveto_reflector = straxen.holoviews_utils.plot_nveto_reflector()

    with TemporaryDirectory() as d:
        # Have to store plot to make sure it is rendered
        hv.save(tpc * diffuer_balls * nveto_reflector, os.path.join(d, 'hitlet_matrix.html'))


def test_pmt_array_plot():
    """Test if PMT array can be plotted for the interactive
    display.
    """
    dummy_gains = np.ones(494)
    dummy_gains[100:120] = 0

    dummy_peak = np.ones(1, dtype=strax.peak_dtype(n_channels=straxen.n_tpc_pmts))

    test = PlotPMTArrayTPC(gains=dummy_gains, top_pmt_array=True)
    test.plot_pmt_array(dummy_peak[0])

    test = PlotPMTArrayTPC(top_pmt_array=False)
    test.plot_pmt_array(dummy_peak[0])

def test_tpc_display_components():
    dummy_peak = np.zeros(1, dtype=strax.peak_dtype(494))
    dummy_peak['dt'] = 1
    dummy_peak['data'][0,:10] = np.arange(10)
    dummy_peak['length'] = 10
    dummy_peak['area_per_channel'] = 1

    peak_plotter = PlotPeaksTPC()
    peak_plotter.plot_peaks(dummy_peak)

    st = straxen.contexts.xenonnt_online(_database_init=False)
    p = st.get_single_plugin('0', 'event_info')
    dummy_events = np.zeros(10, p.dtype)
    inspector = EventStats(dummy_events)
    inspector.interactive_event_stats()
