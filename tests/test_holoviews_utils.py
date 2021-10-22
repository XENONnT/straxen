import strax
import straxen
from straxen.holoviews_utils import nVETOEventDisplay
import holoviews as hv
import panel as pn
import numpy as np
from tempfile import TemporaryDirectory
import os

_dummy_map = straxen.test_utils._nveto_pmt_dummy_df.to_records()


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

    dtype = straxen.veto_events.veto_event_dtype()
    dtype += straxen.veto_events.veto_event_positions_dtype()[2:]
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
    dtype = (straxen.veto_events.veto_event_dtype()
             + straxen.veto_events.veto_event_positions_dtype()[2:])
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
