"""Dynamic holoviews-based waveform display

Note imports are inside function, to keep 'import straxen'
free of holoviews.
"""

import numpy as np
import pandas as pd

import straxen


def seconds_from(t, t_reference):
    return (t - t_reference) / int(1e9)


# Custom wheel zoom tool that only zooms in one dimension
def x_zoom_wheel():
    import bokeh.models
    return bokeh.models.WheelZoomTool(dimensions='width')


@straxen.mini_analysis(requires=['records'], hv_bokeh=True)
def hvdisp_plot_pmt_pattern(*, records, to_pe, array='bottom'):
    """Plot a PMT array, with colors showing the intensity
    of light observed in the time range

    :param array: 'top' or 'bottom', array to show
    """
    import holoviews as hv

    pmts = straxen.pmt_positions()
    areas = np.bincount(records['channel'],
                        weights=records['area'] * to_pe[records['channel']],
                        minlength=len(pmts))

    # Which PMTs should we include?
    m = pmts['array'] == array
    pmts = pmts[m].copy()
    pmts['area'] = areas[m]

    f = 1.08
    pmts = hv.Dataset(
        pmts,
        kdims=[hv.Dimension('x',
                            unit='cm',
                            range=(-straxen.tpc_r * f, straxen.tpc_r * f)),
               hv.Dimension('y',
                            unit='cm',
                            range=(-straxen.tpc_r * f, straxen.tpc_r * f)),
               hv.Dimension('i', label='PMT number'),
               hv.Dimension('area', label='Area', unit='PE')])
    pmts = pmts.to(
        hv.Points,
        vdims=['area', 'i'],
        group='PMTPattern',
        label=array.capitalize()).opts(
        plot=dict(color_index=2,
                  tools=['hover'],
                  show_grid=False),
        style=dict(size=17,
                   cmap='plasma'))

    return pmts


def _records_to_points(*, records, to_pe, t_reference):
    """Return (holoviews.Points, time_stream) corresponding to records
    """
    import holoviews as hv

    areas_r = records['area'] * to_pe[records['channel']]

    # Create dataframe with record metadata
    df = pd.DataFrame(dict(
        area=areas_r,
        time=seconds_from(records['time']
                          + records['dt'] * records['length'] // 2,
                          t_reference),
        channel=records['channel']))

    rec_points = hv.Points(
        df,
        kdims=[hv.Dimension('time', label='Time', unit='sec'),
               hv.Dimension('channel',
                            label='PMT number',
                            range=(0, straxen.n_tpc_pmts))],
        vdims=[hv.Dimension('area', label='Area', unit='pe')])

    time_stream = hv.streams.RangeX(source=rec_points)
    return rec_points, time_stream


@straxen.mini_analysis(requires=['records'], hv_bokeh=True)
def hvdisp_plot_records_2d(records, to_pe,
                           t_reference, width=600, time_stream=None):
    """Plot records in a dynamic 2D histogram of (time, pmt)

    :param width: Plot width in pixels
    :param time_stream: holoviews rangex stream to use. If provided,
    we assume records is already converted to points (which hopefully
    is what the stream is derived from)
    """
    import holoviews as hv
    import holoviews.operation.datashader

    if time_stream is None:
        # Records are still a dataframe, convert it to points
        records, time_stream = _records_to_points(
            records=records, to_pe=to_pe, t_reference=t_reference)

    # TODO: weigh by area?

    return hv.operation.datashader.dynspread(
            hv.operation.datashader.datashade(
                records,
                y_range=(0, straxen.n_tpc_pmts),
                streams=[time_stream])).opts(
        plot=dict(width=width,
                  tools=[x_zoom_wheel(), 'xpan'],
                  default_tools=['save', 'pan', 'box_zoom', 'save', 'reset'],
                  show_grid=False))


@straxen.mini_analysis(
    requires=['peaks', 'peak_basics'],
    hv_bokeh=True)
def hvdisp_plot_peak_waveforms(
        t_reference,
        time_range,
        peaks,
        width=600,
        show_largest=None,
        time_dim=None):
    """Plot the sum waveforms of peaks
    :param width: Plot width in pixels
    :param show_largest: Maximum number of peaks to show
    :param time_dim: Holoviews time dimension; will create new one
    if not provided.
    """
    import holoviews as hv

    if show_largest is not None and len(peaks) > show_largest:
        show_i = np.argsort(peaks['area'])[-show_largest::]
        peaks = peaks[show_i]

    curves = []
    for p in peaks:
        # label = {1: 's1', 2: 's2'}.get(
        #     p['type'], 'unknown')
        color = {1: 'b', 2: 'g'}.get(
            p['type'], 'k')

        # It's better to plot amplitude /time than per bin, since
        # sampling times are now variable
        y = p['data'][:p['length']] / p['dt']
        t_edges = np.arange(p['length'] + 1, dtype=np.int64)
        t_edges = t_edges * p['dt'] + p['time']
        t_edges = seconds_from(t_edges, t_reference)

        # Make a 'step' plot. Unlike matplotlib's steps-mid,
        # this also analyses the final edges correctly
        t_ = np.zeros(2 * len(y))
        y_ = np.zeros(2 * len(y))
        t_[0::2] = t_edges[:-1]
        t_[1::2] = t_edges[1:]
        y_[0::2] = y
        y_[1::2] = y

        if time_dim is None:
            time_dim = hv.Dimension('time', label='Time', unit='sec')

        curves.append(
            hv.Curve(dict(time=t_, amplitude=y_),
                     kdims=time_dim,
                     vdims=hv.Dimension('amplitude', label='Amplitude',
                                        unit='PE/ns'),
                     group='PeakSumWaveform').opts(style=dict(color=color)))

    return hv.Overlay(items=curves).opts(plot=dict(width=width))


def _range_plot(f, full_time_range, t_reference, **kwargs):
    # The **bla is needed to disable some holoviews check
    # on the arguments...
    def wrapped(x_range, **kwargzz):
        if len(kwargzz):
            raise RuntimeError(f"Passed superfluous kwargs {kwargzz}")
        if x_range is None:
            x_range = seconds_from(np.asarray(full_time_range),
                                   t_reference)

        return f(time_range=(t_reference + int(x_range[0] * 1e9),
                             t_reference + int(x_range[1] * 1e9)),
                 t_reference=t_reference,
                 **kwargs)
    return wrapped


@straxen.mini_analysis(
    requires=['records', 'peaks', 'peak_basics'],
    hv_bokeh=True)
def waveform_display(
        context, run_id, to_pe, time_range, t_reference, records, peaks,
        width=600, show_largest=None):
    """Plot a waveform overview display"

    :param width: Plot width in pixels
    """
    import holoviews as hv

    records_points, time_stream = _records_to_points(records=records,
                                                     to_pe=to_pe,
                                                     t_reference=t_reference)

    time_v_channel = context.hvdisp_plot_records_2d(
        run_id=run_id, to_pe=to_pe,
        records=records_points,
        width=width,
        time_stream=time_stream,
        time_range=time_range, t_reference=t_reference,
        # We don't need to cut these further, records we get are already cut
        # to the plot's maximum range by the mini_analysis logic
        # and datashader does the online cutting / rebinning / zooming.
        # This is fortunate, since we omitted 'endtime' from records_points!
        time_selection='skip')

    array_plot = {
        array: hv.DynamicMap(
            _range_plot(
                context.hvdisp_plot_pmt_pattern,
                run_id=run_id, to_pe=to_pe,
                records=records,
                full_time_range=time_range,
                t_reference=t_reference,
                time_selection='touching',
                array=array),
            streams=[time_stream])
        for array in ('top', 'bottom')}

    peak_wvs = hv.DynamicMap(
        _range_plot(
            context.hvdisp_plot_peak_waveforms,
            run_id=run_id,
            width=width,
            full_time_range=time_range,
            t_reference=t_reference,
            time_selection='touching',
            time_dim=records_points.kdims[0],
            peaks=peaks,
            show_largest=show_largest),
        streams=[time_stream])

    layout = (peak_wvs + array_plot['top']
              + time_v_channel + array_plot['bottom'])
    return layout.cols(2)
