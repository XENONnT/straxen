"""Dynamic holoviews-based waveform display
Note imports are inside function, to keep 'import straxen'
free of holoviews.
"""

import numpy as np
import pandas as pd

import straxen

straxen._BOKEH_X_RANGE = None


def seconds_from(t, t_reference, unit_conversion=int(1e9)):
    """
    Helper function which concerts times into relative times in
    specified unit.

    :param t: Time
    :param t_reference: Reference time e.g. start of an event or first
        peak in event.
    :param unit_conversion: Conversion factor for time units e.g. 10**3
        for micro seconds.
    """
    return (t - t_reference) / unit_conversion


# Custom wheel zoom tool that only zooms in one dimension
def x_zoom_wheel():
    import bokeh.models
    return bokeh.models.WheelZoomTool(dimensions='width')


@straxen.mini_analysis(requires=['records'], hv_bokeh=True)
def hvdisp_plot_pmt_pattern(*, config, records, to_pe, array='bottom'):
    """Plot a PMT array, with colors showing the intensity
    of light observed in the time range
    :param array: 'top' or 'bottom', array to show
    """
    import holoviews as hv

    pmts = straxen.pmt_positions(xenon1t=config['n_tpc_pmts'] < 300)
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
               hv.Dimension('i', range=(0, config['n_tpc_pmts']), label='PMT number'),
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


def _records_to_points(*, records, to_pe, t_reference,
                       config, unit_conversion=int(1e9)):
    """Return (holoviews.Points, time_stream) corresponding to records.
    """
    import holoviews as hv

    areas_r = records['area'] * to_pe[records['channel']]

    # Create dataframe with record metadata
    df = pd.DataFrame(dict(
        area=areas_r,
        time=seconds_from(records['time'] + records['dt'] * records['length'] // 2,
                          t_reference, unit_conversion=unit_conversion),
        channel=records['channel']))

    rec_points = hv.Points(
        df,
        kdims=[hv.Dimension('time', label='Time [µs]'),
               hv.Dimension('channel',
                            label='PMT Channel',
                            range=(0, config['n_tpc_pmts']))],
        vdims=[hv.Dimension('area', label='Area [pe]')])

    time_stream = hv.streams.RangeX(source=rec_points)
    return rec_points, time_stream


@straxen.mini_analysis(requires=['records'], hv_bokeh=True)
def hvdisp_plot_records_2d(records,
                           to_pe,
                           config,
                           t_reference,
                           time_stream=None,
                           tools=(x_zoom_wheel(), 'xpan'),
                           default_tools=('save', 'pan', 'box_zoom', 'save', 'reset'),
                           plot_library='bokeh',
                           hooks=()):
    """Plot records in a dynamic 2D histogram of (time, pmt)
    :param width: Plot width in pixels
    :param time_stream: holoviews rangex stream to use. If provided,
        we assume records is already converted to points (which hopefully
        is what the stream is derived from)
    :param tools: Tools to be used in the interactive plot. Only works
        with bokeh as plot library.
    :param plot_library: Default bokeh, library to be used for the
        plotting.
    :param width: With of the record matrix in pixel.
    :param hooks: Hooks to adjust plot settings.
    :returns: datashader object, records holoview points,
        RangeX time stream of records.
    """
    shader, records, time_stream = _hvdisp_plot_records_2d(records,
                                                           to_pe,
                                                           config,
                                                           t_reference,
                                                           time_stream=time_stream,
                                                           default_tools=default_tools,
                                                           tools=tools,
                                                           hooks=hooks,
                                                           plot_library=plot_library)
    shader = shader.opts(title="Time vs. Channel")
    return shader


def _hvdisp_plot_records_2d(records,
                            to_pe,
                            config,
                            t_reference,
                            event_range=(None, None),
                            time_stream=None,
                            default_tools=(),
                            tools=(),
                            hooks=(),
                            plot_library='bokeh',
                            unit_conversion=10**3):
    import holoviews as hv
    import holoviews.operation.datashader
    hv.extension(plot_library)

    if time_stream is None:
        # Records are still a dataframe, convert it to points
        records, time_stream = _records_to_points(records=records,
                                                  to_pe=to_pe,
                                                  t_reference=t_reference,
                                                  unit_conversion=unit_conversion,
                                                  config=config)

    # Whether to show the toolbar or not:
    if tools:
        toolbar = 'right'
    else:
        toolbar = None

    # Creating the plot:
    shader = hv.operation.datashader.dynspread(
        hv.operation.datashader.datashade(
            records,
            dynamic=True,
            x_range=event_range,
            y_range=(0, config['n_tpc_pmts']),
            streams=[time_stream]), threshold=0.1).opts(
        plot=dict(aspect=4,
                  responsive='width',
                  hooks=list(hooks),
                  toolbar=toolbar,
                  tools=list(tools),
                  default_tools=list(default_tools),
                  fontsize={'labels': 12},
                  show_grid=True))

    return shader, records, time_stream


def plot_record_polygons(record_points, width=1.1, **kwargs):
    """
    Plots record hv.Points as polygons for record matrix.

    :param record_points: Holoviews Points generated with
        _records_to_points.
    :param width: Length of the record in µs.
    :param kwargs: Keyword arguments applied to hv.Polygons options.
    :returns: hv.Polygons
    """
    import holoviews as hv

    data = [{('x', 'y'): rectangle(t, ch, width=width), 'area': a} for t, ch, a in
            record_points.data[['time', 'channel', 'area']].values]
    polys = hv.Polygons(data, vdims='area')
    polys = polys.opts(color='level', aspect=4, responsive='width', line_width=0, **kwargs, cmap='viridis')
    return polys


def rectangle(time=0, channel=0, width=1.1, height=1):
    """
    Generates polygon box coordinates for record matrix.

    :param time: Center position of the record in time.
    :param channel: Center position of PMT channel. E.g channel 0 => 0.5
    :param width: Length of the record in µs.
    param height: Height of the record in "channel"-units.

    X,Y have to be the center of the polygon.
    Width and height are the full width and height in data coordinates.
    """
    width = width / 2
    height = height / 2
    return np.array([(time - width, channel - height),
                     (time + width, channel - height),
                     (time + width, channel + height),
                     (time - width, channel + height)])


def get_records_matrix_in_window(polys, x_range, time_slice=10):
    """
    Helper function which returns polygons for rendering when x_range
    is below the specified value.

    This function has to be applied to polygons e.g.:

        poly.apply(get_records_matrix_in_window, streams=[time_stream])

    :param polys: Holoviews Polygons
    :param x_range: x_range of the RangeX object.
    :param time_slice: Size of the time slice in [µs] when records
        should be drawn.
    """
    if x_range is None:
        # Needed since x_range is by default not defined when plotting
        # the first time.
        return polys.iloc[:0]
    if (x_range[1] - x_range[0]) < time_slice:
        # If x_range smaller than specified minimum return polys ->
        # render polys.
        inds = _in_window(polys.data, x_range)
        return polys.iloc[inds]
    return polys.iloc[:0]


def _in_window(polys, x_range):
    """
    Function which checks if a polygon is partially in x_range.

    :param polys: List of ordered dictionaries containing Polygon data.
    :param x_range: Range which should be tested.
    """
    res = []
    for ind, p in enumerate(polys):
        # Loop over polys, if partially in window keep index.
        x = p['x']
        if np.any((x_range[0] <= x) & (p['x'] < x_range[1])):
            res.append(ind)
    return res


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

        # Deal with strange time ranges -- not sure how these arise?
        x_range = np.nan_to_num(x_range)
        if x_range[1] == x_range[0]:
            x_range[1] += 1

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
        config,
        width=600, show_largest=None):
    """Plot a waveform overview display"
    :param width: Plot width in pixels
    """
    import holoviews as hv

    records_points, time_stream = _records_to_points(records=records,
                                                     to_pe=to_pe,
                                                     t_reference=t_reference,
                                                     config=config)

    time_v_channel = context.hvdisp_plot_records_2d(
        run_id=run_id, to_pe=to_pe,
        records=records_points,
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

    layout = time_v_channel + peak_wvs + array_plot['top'] + array_plot['bottom']
    return layout.cols(2)


def hook(plot, x_range, debug=False):
    """
    Hook to set the same RangeX stream for event display and
    records matrix, voodoo....

    Note:
        Works only in the following order:

        1. Create holoviews
        2. hv.render plot
        3. set bokeh x_range as holoviews x_range

        Does not work first with bokeh and then with holoviews. Why?
        I have no clue....
    """
    if debug:
        print('Hook: ', x_range)
    if not x_range:
        _hook_get_xrange(plot, debug)
    else:
        _hook_set_xrange(plot, x_range, debug)


def _hook_get_xrange(plot, debug):
    straxen._BOKEH_X_RANGE = plot.handles['x_range']
    if debug:
        print('Get', straxen._BOKEH_X_RANGE)


def _hook_set_xrange(plot, x_range, debug):
    if debug:
        print('Set', x_range)
    plot.state.x_range = x_range
