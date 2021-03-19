import bokeh
import bokeh.plotting as bklt

from straxen.analyses.holoviews_waveform_display import _hvdisp_plot_records_2d, hook, \
    plot_record_polygons, get_records_matrix_in_window

import numpy as np
import numba
import strax
import straxen

import warnings

# Default legend, unknow, S1 and S2
LEGENDS = ('Unknown', 'S1', 'S2')
straxen._BOKEH_CONFIGURED_NOTEBOOK = False


@straxen.mini_analysis(requires=('events', 'event_basics', 'peaks', 'peak_basics', 'peak_positions'),
                       warn_beyond_sec=0.05)
def event_display_interactive(events,
                              peaks,
                              to_pe,
                              run_id,
                              context,
                              bottom_pmt_array=True,
                              only_main_peaks=False,
                              only_peak_detail_in_wf=False,
                              plot_all_pmts=False,
                              plot_record_matrix=False,
                              plot_records_threshold=10,
                              xenon1t=False,
                              colors=('gray', 'blue', 'green'),
                              yscale=('linear', 'linear', 'linear'),
                              log=True, ):
    """
    Interactive event display for XENONnT. Plots detailed main/alt
    S1/S2, bottom and top PMT hit pattern as well as all other peaks
    in a given event.

    :param bottom_pmt_array: If true plots bottom PMT array hit-pattern.
    :param only_main_peaks: If true plots only main peaks into detail
        plots as well as PMT arrays.
    :param only_peak_detail_in_wf: Only plots main/alt S1/S2 into
        waveform. Only plot main peaks if only_main_peaks is true.
    :param plot_all_pmts: Bool if True, colors switched off PMTs instead
        of showing them in gray, useful for graphs shown in talks.
    :param plot_record_matrix: If true record matrix is plotted below.
        waveform.
    :param plot_records_threshold: Threshold at which zoom level to display
        record matrix as polygons. Larger values may lead to longer
        render times since more polygons are shown.
    :param xenon1t: Flag to use event display with 1T data.
    :param colors: Colors to be used for peaks. Order is as peak types,
        0 = Unknown, 1 = S1, 2 = S2. Can be any colors accepted by bokeh.
    :param yscale: Defines scale for main/alt S1 == 0, main/alt S2 == 1,
        waveform plot == 2. Please note, that the log scale can lead to funny
        glyph renders for small values.
    :param log: If true color sclae is used for hitpattern plots.

    Note:
        How to use:

        > st.event_display(context,
        >                  run_id,
        >                  time_range=(event['time'],
        >                              event['endtime'])
        >                  )

    Warning:
        Raises an error if the user queries a time range which contains
        more than a single event.

    :return: bokeh.plotting.figure instance.
    """
    st = context

    if len(yscale) != 3:
        raise ValueError(f'"yscale" needs three entries, but you passed {len(yscale)}.')

    if not hasattr(st, '_BOKEH_CONFIGURED_NOTEBOOK'):
        st._BOKEH_CONFIGURED_NOTEBOOK = True
        # Configure show to show notebook:
        from bokeh.io import output_notebook
        output_notebook()

    if len(events) != 1:
        raise ValueError('The time range you specified contains more or'
                         ' less than a single event. The event display '
                         ' only works with individual events for now.')

    if peaks.shape[0] == 0:
        raise ValueError('Found an event without peaks this should not had have happened.')

    # Select main/alt S1/S2s based on time and endtime in event:
    m_other_peaks = np.ones(len(peaks), dtype=np.bool_)  # To select non-event peaks
    endtime = strax.endtime(peaks)

    signal = {}
    if only_main_peaks:
        s1_keys = ['s1']
        s2_keys = ['s2']
        labels = {'s1': 'S1', 's2': 'S2'}
    else:
        s1_keys = ['s1', 'alt_s1']
        s2_keys = ['s2', 'alt_s2']
        labels = {'s1': 'MS1', 'alt_s1': 'AS1', 's2': 'MS2', 'alt_s2': 'AS2'}

    for s_x in labels.keys():
        # Loop over Main/Alt Sx and get store S1/S2 Main/Alt in signals,
        # store information about other peaks as "m_other_peaks"
        m = (peaks['time'] == events[f'{s_x}_time']) & (endtime == events[f'{s_x}_endtime'])
        signal[s_x] = peaks[m]
        m_other_peaks &= ~m

    # Detail plots for main/alt S1/S2:
    fig_s1, fig_s2 = plot_detail_plot_s1_s2(signal,
                                            s1_keys,
                                            s2_keys,
                                            labels,
                                            colors,
                                            yscale[:2], )

    # PMT arrays:
    if not only_main_peaks:
        # Plot all keys into both arrays:
        top_array_keys = s2_keys + s1_keys
        bottom_array_keys = s1_keys + s2_keys
    else:
        top_array_keys = s2_keys
        bottom_array_keys = s1_keys

    fig_top, fig_bottom = plot_pmt_arrays_and_positions(top_array_keys,
                                                        bottom_array_keys,
                                                        signal,
                                                        to_pe,
                                                        labels,
                                                        plot_all_pmts,
                                                        xenon1t=xenon1t,
                                                        log=log)

    m_other_s2 = m_other_peaks & (peaks['type'] == 2)
    if np.any(m_other_s2) and not only_main_peaks:
        # Now we have to add the positions of all the other S2 to the top pmt array
        # if not only main peaks.
        fig_top, plot = plot_posS2s(peaks[m_other_s2], label='OS2s', fig=fig_top, s2_type_style_id=2)
        plot.visible = False

    # Main waveform plot:
    if only_peak_detail_in_wf:
        # If specified by the user only plot main/alt S1/S2
        peaks = peaks[~m_other_peaks]

    waveform = plot_event(peaks, signal, labels, events[0], colors, yscale[-1])

    # Create tile:
    title = _make_event_title(events[0], run_id)

    # Put everything together:
    if bottom_pmt_array:
        upper_row = [fig_s1, fig_s2, fig_top, fig_bottom]
    else:
        upper_row = [fig_s1, fig_s2, fig_top]

    upper_row = bokeh.layouts.Row(children=upper_row)

    plots = bokeh.layouts.gridplot(children=[upper_row, waveform],
                                   sizing_mode='scale_both',
                                   ncols=1,
                                   merge_tools=True,
                                   toolbar_location='above',
                                   )
    event_display = bokeh.layouts.Column(children=[title, plots],
                                         sizing_mode='scale_both',
                                         max_width=1600,
                                         )

    # Add record matrix if asked:
    if plot_record_matrix:
        if st.is_stored(run_id, 'records'):
            # Check if records can be found and load:
            r = st.get_array(run_id, 'records', time_range=(events[0]['time'], events[0]['endtime']))
        elif st.is_stored(run_id, 'raw_records'):
            warnings.warn(f'Cannot find records for {run_id}, making them from raw_records instead.')
            p = st.get_single_plugin(run_id, 'records')
            r = st.get_array(run_id, 'raw_records', time_range=(events[0]['time'], events[0]['endtime']))
            r = p.compute(r, events[0]['time'], events[0]['endtime'])['records']
        else:
            warnings.warn(f'Can neither find records nor raw_records for run {run_id}, proceed without record '
                          f'matrix.')
            plot_record_matrix = False

    if plot_record_matrix:
        straxen._BOKEH_X_RANGE = None
        # First get hook to for x_range:
        x_range_hook = lambda plot, element: hook(plot, x_range=straxen._BOKEH_X_RANGE, debug=False)

        # Create datashader plot:
        wf, record_points, time_stream = _hvdisp_plot_records_2d(records=r,
                                                                 to_pe=to_pe,
                                                                 t_reference=peaks[0]['time'],
                                                                 event_range=(waveform.x_range.start,
                                                                              waveform.x_range.end),
                                                                 config=st.config,
                                                                 hooks=[x_range_hook],
                                                                 tools=[],
                                                                 )
        # Create record polygons:
        polys = plot_record_polygons(record_points, width=1.1)
        records_in_window = polys.apply(get_records_matrix_in_window,
                                        streams=[time_stream],
                                        time_slice=plot_records_threshold)

        # Render plot to initialize x_range:
        import holoviews as hv
        import panel

        _ = hv.render(wf)
        # Set x-range of event plot:
        bokeh_set_x_range(waveform, straxen._BOKEH_X_RANGE, debug=False)
        event_display = panel.Column(event_display,
                                     wf * records_in_window,
                                     sizing_mode='scale_width')

    return event_display


def plot_detail_plot_s1_s2(signal,
                           s1_keys,
                           s2_keys,
                           labels,
                           colors,
                           yscale=('linear', 'linear')):
    """
    Function to plot the main/alt S1/S2 peak details.

    :param signal: Dictionary containing the peak information.
    :param s1_keys: S1 keys to be plotted e.g. with and without alt S1
    :param s2_keys: Same but for S2
    :param labels: Labels to be used for Peaks
    :param colors: Colors to be used
    :param yscale: Tuple with axis scale type.

    :return: S1 and S2 bokeh figure.
    """
    # First we create figure then we loop over figures and plots and
    # add drawings:
    fig_s1 = straxen.bokeh_utils.default_fig(title='Main/Alt S1',
                                             y_axis_type=yscale[0], )
    fig_s2 = straxen.bokeh_utils.default_fig(title='Main/Alt S2',
                                             y_axis_type=yscale[1], )

    for fig, peak_types in zip([fig_s1, fig_s2],
                               (s1_keys, s2_keys)):
        # Loop over fig and corresponding peak keys
        for peak_type in peak_types:
            if 's2' in peak_type:
                # If S2 use µs as units
                time_scalar = 1000  # ns
                unit = 'µs'
            else:
                time_scalar = 1  # ns
                unit = 'ns'
            if signal[peak_type].shape[0]:
                # If signal exists, plot:
                fig, plot = plot_peak_detail(signal[peak_type],
                                             time_scalar=time_scalar,
                                             label=labels[peak_type],
                                             unit=unit,
                                             fig=fig,
                                             colors=colors)
                if 'alt' in peak_type:
                    # Not main S1/S2, so make peak invisible
                    plot.visible = False
    return fig_s1, fig_s2


def plot_pmt_arrays_and_positions(top_array_keys,
                                  bottom_array_keys,
                                  signal,
                                  to_pe,
                                  labels,
                                  plot_all_pmts,
                                  xenon1t=False,
                                  log=True):
    """
    Function which plots the Top and Bottom PMT array.

    :returns: fig_top, fig_bottom
    """
    # Same logic as for detailed Peaks, first make figures
    # then loop over figures and data and populate figures with plots
    fig_top = straxen.bokeh_utils.default_fig(title='Top array')
    fig_bottom = straxen.bokeh_utils.default_fig(title='Bottom array')

    for pmt_array_type, fig, peak_types in zip(['top', 'bottom'],
                                               [fig_top, fig_bottom],
                                               [top_array_keys, bottom_array_keys]):
        for ind, k in enumerate(peak_types):
            # Loop over peaks enumerate them since we plot all Peaks
            # Main/ALt S1/S2 into the PMT array, but only the first one
            # Should be visible.
            if not signal[k].shape[0]:
                # alt S1/S2 does not exist so go to next.
                continue

            fig, plot, _ = plot_pmt_array(signal[k][0],
                                          pmt_array_type, to_pe,
                                          plot_all_pmts=plot_all_pmts,
                                          label=labels[k],
                                          xenon1t=xenon1t,
                                          fig=fig,
                                          log=log)
            if ind:
                # Not main S1 or S2
                plot.visible = False

            if pmt_array_type == 'top' and 's2' in k:
                # In case of the top PMT array we also have to plot the S2 positions:
                fig, plot = plot_posS2s(signal[k][0], label=labels[k], fig=fig, s2_type_style_id=ind)
                if ind:
                    # Not main S2
                    plot.visible = False

    return fig_top, fig_bottom


def plot_event(peaks, signal, labels, event, colors, yscale='linear'):
    """
    Wrapper for plot peaks to highlight main/alt. S1/S2

    :param peaks: Peaks in event
    :param signal: Dictionary containing main/alt. S1/S2
    :param labels: dict with labels to be used
    :param event: Event to set correctly x-ranges.
    :param colors: Colors to be used for unknown, s1 and s2 signals.
    :param yscale: string of yscale type.

    :return: bokeh.plotting.figure instance
    """
    waveform = plot_peaks(peaks, time_scalar=1000, colors=colors, yscale=yscale)
    # Highlight main and alternate S1/S2:
    start = peaks[0]['time']
    end = strax.endtime(peaks)[-1]
    # Workaround did not manage to scale via pixels...
    ymax = np.max((peaks['data'].T / peaks['dt']).T)
    ymax -= 0.1 * ymax
    for s, p in signal.items():
        if p.shape[0]:
            pos = (p[0]['center_time'] - start) / 1000
            main = bokeh.models.Span(location=pos,
                                     dimension='height',
                                     line_alpha=0.6,
                                     )
            vline_label = bokeh.models.Label(x=pos,
                                             y=ymax,
                                             angle=np.pi / 2,
                                             text=labels[s],
                                             )
            if 'alt' in s:
                main.line_dash = 'dotted'
            else:
                main.line_dash = 'dashed'
            waveform.add_layout(main)
            waveform.add_layout(vline_label)

    # Get some meaningful x-range limit to 10% left and right extending
    # beyond first last peak, clip at event boundary.
    length = (end - start) / 10**3

    waveform.x_range.start = max(-0.1 * length, (event['time'] - start) / 10**3)
    waveform.x_range.end = min(1.1 * length, (event['endtime'] - start) / 10**3)
    return waveform


def plot_peak_detail(peak,
                     time_scalar=1,
                     label='',
                     unit='ns',
                     colors=('gray', 'blue', 'green'),
                     fig=None, ):
    """
    Function which makes a detailed plot for the given peak. As in the
    main/alt S1/S2 plots of the event display.

    :param peak: Peak to be plotted.
    :param time_scalar: Factor to rescale the time from ns to other scale.
        E.g. =1000 scales to µs.
    :param label: Label to be used in the plot legend.
    :param unit: Time unit of the plotted peak.
    :param colors: Colors to be used for unknown, s1 and s2 peaks.
    :param fig: Instance of bokeh.plotting.figure if None one will be
        created via straxen.bokeh.utils.default_figure().
    :return: Instance of bokeh.plotting.figure
    """
    if not peak.shape:
        peak = np.array([peak])

    if peak.shape[0] != 1:
        raise ValueError('Cannot plot the peak details for more than one '
                         'peak. Please make sure peaks has the shape (1,)!')

    p_type = peak[0]['type']

    if not fig:
        fig = straxen.bokeh_utils.default_fig(title=f'Main/Alt S{p_type}')

    tt = straxen.bokeh_utils.peak_tool_tip(p_type)
    tt = [v for k, v in tt.items() if k not in ['time_static', 'center_time', 'endtime']]
    fig.add_tools(bokeh.models.HoverTool(names=[label],
                                         tooltips=tt)
                  )

    source = straxen.bokeh_utils.get_peaks_source(peak,
                                                  relative_start=peak[0]['time'],
                                                  time_scaler=time_scalar,
                                                  keep_amplitude_per_sample=False
                                                  )

    patches = fig.patches(source=source,
                          legend_label=label,
                          fill_color=colors[p_type],
                          fill_alpha=0.2,
                          line_color=colors[p_type],
                          line_width=0.5,
                          name=label
                          )
    fig.xaxis.axis_label = f"Time [{unit}]"
    fig.xaxis.axis_label_text_font_size = '12pt'
    fig.yaxis.axis_label = "Amplitude [pe/ns]"
    fig.yaxis.axis_label_text_font_size = '12pt'

    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"

    if label:
        fig.legend.visible = True
    else:
        fig.legend.visible = False

    return fig, patches


def plot_peaks(peaks, time_scalar=1, fig=None, colors=('gray', 'blue', 'green'), yscale='linear'):
    """
    Function which plots a list/array of peaks relative to the first
    one.

    :param peaks: Peaks to be plotted.
    :param time_scalar:  Factor to rescale the time from ns to other scale.
        E.g. =1000 scales to µs.
    :param colors: Colors to be used for unknown, s1 and s2 signals
    :param yscale: yscale type can be "linear" or "log"
    :param fig: Instance of bokeh.plotting.figure if None one will be
        created via straxen.bokeh.utils.default_figure().
    :return: bokeh.plotting.figure instance.
    """
    if not fig:
        fig = straxen.bokeh_utils.default_fig(width=1600, height=400, y_axis_type=yscale)

    for i in range(0, 3):
        _ind = np.where(peaks['type'] == i)[0]
        if not len(_ind):
            continue

        source = straxen.bokeh_utils.get_peaks_source(peaks[_ind],
                                                      relative_start=peaks[0]['time'],
                                                      time_scaler=time_scalar,
                                                      keep_amplitude_per_sample=False
                                                      )

        fig.patches(source=source,
                    fill_color=colors[i],
                    fill_alpha=0.2,
                    line_color=colors[i],
                    line_width=0.5,
                    legend_label=LEGENDS[i],
                    name=LEGENDS[i],
                    )

        tt = straxen.bokeh_utils.peak_tool_tip(i)
        tt = [v for k, v in tt.items() if k != 'time_dynamic']
        fig.add_tools(bokeh.models.HoverTool(names=[LEGENDS[i]], tooltips=tt))
        fig.add_tools(bokeh.models.WheelZoomTool(dimensions='width', name='wheel'))
        fig.toolbar.active_scroll = [t for t in fig.tools if t.name == 'wheel'][0]

    fig.xaxis.axis_label = 'Time [µs]'
    fig.xaxis.axis_label_text_font_size = '12pt'
    fig.yaxis.axis_label = "Amplitude [pe/ns]"
    fig.yaxis.axis_label_text_font_size = '12pt'

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    return fig


def plot_pmt_array(peak,
                   array_type,
                   to_pe,
                   plot_all_pmts=False,
                   log=False,
                   xenon1t=False,
                   fig=None,
                   label='', ):
    """
    Plots top or bottom PMT array for given peak.

    :param peak: Peak for which the hit pattern should be plotted.
    :param array_type: String which specifies if "top" or "bottom"
        PMT array should be plotted
    :param to_pe: PMT gains.
    :param log: If true use a log-scale for the color scale.
    :param plot_all_pmts: If True colors all PMTs instead of showing
        swtiched off PMTs as gray dots.
    :param xenon1t: If True plots 1T array.
    :param fig: Instance of bokeh.plotting.figure if None one will be
        created via straxen.bokeh.utils.default_figure().
    :param label: Label of the peak which should be used for the
        plot legend

    :returns: Tuple containing a bokeh figure, glyph and transform
        instance.
    """
    if peak.shape:
        raise ValueError('Can plot PMT array only for a single peak at a time.')

    tool_tip = [('Plot', '$name'),
                ('Channel', '@pmt'),
                ('X-Position [cm]', '$x'),
                ('Y-Position [cm]', '$y'),
                ('area [pe]', '@area')
                ]

    array = ('top', 'bottom')
    if array_type not in array:
        raise ValueError('"array_type" must be either top or bottom.')

    if not fig:
        fig = straxen.bokeh_utils.default_fig(title=f'{array_type} array')

    # Creating TPC axis and title
    fig = _plot_tpc(fig)

    # Plotting PMTs:
    pmts = straxen.pmt_positions(xenon1t)
    if plot_all_pmts:
        mask_pmts = np.zeros(len(pmts), dtype=np.bool_)
    else:
        mask_pmts = to_pe == 0
    pmts_on = pmts[~mask_pmts]
    pmts_on = pmts_on[pmts_on['array'] == array_type]

    if np.any(mask_pmts):
        pmts_off = pmts[mask_pmts]
        pmts_off = pmts_off[pmts_off['array'] == array_type]
        fig = _plot_off_pmts(pmts_off, fig)

    area_per_channel = peak['area_per_channel'][pmts_on['i']]

    if log:
        area_plot = np.log10(area_per_channel)
        # Manually set infs to zero since cmap cannot handle it.
        area_plot = np.where(area_plot == -np.inf, 0, area_plot)
    else:
        area_plot = area_per_channel

    mapper = bokeh.transform.linear_cmap(field_name='area_plot',
                                         palette="Viridis256",
                                         low=min(area_plot),
                                         high=max(area_plot))

    source_on = bklt.ColumnDataSource(data={'x': pmts_on['x'],
                                            'y': pmts_on['y'],
                                            'area': area_per_channel,
                                            'area_plot': area_plot,
                                            'pmt': pmts_on['i']
                                            }
                                      )

    p = fig.scatter(source=source_on,
                    radius=straxen.tpc_pmt_radius,
                    fill_color=mapper,
                    fill_alpha=1,
                    line_color='black',
                    legend_label=label,
                    name=label + '_pmt_array',
                    )
    fig.add_tools(bokeh.models.HoverTool(names=[label + '_pmt_array'], tooltips=tool_tip))
    fig.legend.location = 'top_left'
    fig.legend.click_policy = "hide"
    fig.legend.orientation = "horizontal"
    fig.legend.padding = 0
    fig.toolbar_location = None
    return fig, p, mapper


def _plot_tpc(fig=None):
    """
    Plots ring at TPC radius and sets xy limits + labels.
    """
    if not fig:
        fig = straxen.bokeh_utils.default_fig()

    fig.circle(x=0, y=0,
               radius=straxen.tpc_r,
               fill_color="white",
               line_color="black",
               line_width=3,
               fill_alpha=0
               )
    fig.xaxis.axis_label = 'x [cm]'
    fig.xaxis.axis_label_text_font_size = '12pt'
    fig.yaxis.axis_label = 'y [cm]'
    fig.yaxis.axis_label_text_font_size = '12pt'
    fig.x_range.start = -80
    fig.x_range.end = 80
    fig.y_range.start = -80
    fig.y_range.end = 80

    return fig


def _plot_off_pmts(pmts, fig=None):
    """
    Plots PMTs which are switched off.
    """
    if not fig:
        fig = straxen.bokeh_utils.default_fig()
    fig.circle(x=pmts['x'],
               y=pmts['y'],
               fill_color='gray',
               line_color='black',
               radius=straxen.tpc_pmt_radius,
               )
    return fig


def plot_posS2s(peaks, label='', fig=None, s2_type_style_id=0):
    """
    Plots xy-positions of specified peaks.

    :param peaks: Peaks for which the position should be plotted.
    :param label: Legend label and plot name (name serves as idenitfier).
    :param fig: bokeh.plotting.figure instance the plot should be plotted
        into. If None creates new instance.
    :param s2_type_style_id: 0 plots main S2 style, 1 for alt S2 and
        2 for other S2s (e.g. single electrons).
    """
    if not peaks.shape:
        peaks = np.array([peaks])

    if not np.all(peaks['type'] == 2):
        raise ValueError('All peaks must be S2!')

    if not fig:
        fig = straxen.bokeh_utils.default_fig()

    source = straxen.bokeh_utils.get_peaks_source(peaks)

    if s2_type_style_id == 0:
        p = fig.cross(source=source,
                      name=label,
                      legend_label=label,
                      color='red',
                      line_width=2,
                      size=12)

    if s2_type_style_id == 1:
        p = fig.cross(source=source,
                      name=label,
                      legend_label=label,
                      color='orange',
                      angle=45 / 360 * 2 * np.pi,
                      line_width=2,
                      size=12)

    if s2_type_style_id == 2:
        p = fig.diamond_cross(source=source,
                              name=label,
                              legend_label=label,
                              color='red',
                              size=8
                              )

    tt = straxen.bokeh_utils.peak_tool_tip(2)
    tt = [v for k, v in tt.items() if k not in ['time_dynamic', 'amplitude']]
    fig.add_tools(bokeh.models.HoverTool(names=[label],
                                         tooltips=[('Position x [cm]', '@x'),
                                                   ('Position y [cm]', '@y')] + tt))
    return fig, p


def _make_event_title(event, run_id, width=1600):
    """
    Function which makes the title of the plot for the specified event.

    Note:
        To center the title I use a transparent box.

    :param event: Event which we are plotting
    :param run_id: run_id

    :returns: Title as bokeh.models.Div instance
    """
    start = event['time']
    date = np.datetime_as_string(start.astype('<M8[ns]'), unit='s')
    start_ns = start - (start // 10**9) * 10**9
    end = strax.endtime(event)
    end_ns = end - start + start_ns
    event_number = event['event_number']
    text = (f'<h2>Event {event_number} from run {run_id}<br>'
            f'Recorded at {date[:10]} {date[10:]} UTC,'
            f' {start_ns} ns - {end_ns} ns </h2>')

    title = bokeh.models.Div(text=text,
                             style={'text-align': 'left',
                                    },
                             sizing_mode='scale_both',
                             width=width,
                             default_size=width,
                             orientation='vertical',
                             width_policy='fit',
                             margin=(0, 0, -30, 50)
                             )
    return title


def bokeh_set_x_range(plot, x_range, debug=False):
    """
    Function which adjust java script call back for x_range of a bokeh
    plot. Required to link bokeh and holoviews x_range.

    Note:
        This is somewhat voodoo + some black magic,
        but it works....
    """
    from bokeh.models import CustomJS
    code = """\
    const start = cb_obj.start;
    const end = cb_obj.end;
    // Need to update the attributes at the same time.
    x_range.setv({start, end});
    """
    for attr in ['start', 'end']:
        if debug:
            # Prints x_range bar to check Id, as I said voodoo
            print(x_range)
        plot.x_range.js_on_change(attr, CustomJS(args=dict(x_range=x_range), code=code))


class DataSelectionHist:
    """
    Class for an interactive data selection plot.
    """
    def __init__(self, name, size=600):
        """
         Class for an interactive data selection plot.

        :param name: Name of the class object instance. Needed for
            dynamic return, e.g. ds = DataSelectionHist("ds")
        :param size: Edge size of the figure in pixel.
        """
        self.name = name
        self.selection_index = None
        self.size = size

    def histogram2d(self,
                    items,
                    xdata,
                    ydata,
                    bins,
                    hist_range,
                    x_label='X-Data',
                    y_label='Y-Data',
                    log_color_scale=True,
                    cmap_steps=256,
                    clim=(None, None),
                    undeflow_color=None,
                    overflow_color=None,
                    weights=1):
        """
        2d Histogram which allows to select the plotted items dynamically.

        Note:
            You can select the data either via a box select or Lasso
            select tool. The data can be returned by:

            ds.get_back_selected_items()

            Hold shift to select multiple regions.

        Warnings:
            Depending on the number of bins the Lasso selection can
            become relatively slow. The number of bins should not be
            larger than 100.
            The box selection performance is better.

        :param items: numpy.structured.array of items to be selected.
            e.g. peaks or events.
        :param xdata: numpy.array for xdata e.g. peaks['area']
        :param ydata: same
        :param bins: Integer specifying the number of bins. Currently
            x and y axis must share the same binning.
        :param hist_range: Tuple of x-range and y-range.
        :param x_label: Label to be used for the x-axis
        :param y_label: same but for y
        :param log_color_scale: If true (default) use log colorscale
        :param cmap_steps: Integer between 0 and 256 for stepped
            colorbar.
        :param clim: Tuple of color limits.
        :param undeflow_color: If specified colors all bins below clim
            with the corresponding color.
        :param overflow_color: Same but per limit.
        :param weights: If specified each bin entry is weighted by this
            value. Can be either a scalar e.g. a time or an array of
            weights which has the same length as the x/y data.
        :return: bokeh figure instance.
        """
        if isinstance(bins, tuple):
            raise ValueError('Currently only squared bins are supported. '
                             'Plase change bins into an integer.')

        x_pos, y_pos = self._make_bin_positions((bins, bins), hist_range)
        weights = np.ones(len(xdata)) * weights

        hist, hist_inds = self._hist2d_with_index(xdata,
                                                  ydata,
                                                  weights,
                                                  self.xedges,
                                                  self.yedges)

        # Define times and ids for return:
        self.items = items
        self.hist_inds = hist_inds

        colors = self._get_color(hist,
                                 cmap_steps,
                                 log_color_scale=log_color_scale,
                                 clim=clim,
                                 undeflow_color=undeflow_color,
                                 overflow_color=overflow_color)

        # Create Figure and add LassoTool:
        f = bokeh.plotting.figure(title="DataSelection",
                                  width=self.size,
                                  height=self.size,
                                  tools="box_select,reset,save")

        # Add hover tool, colorbar is too complictaed:
        tool_tip = [("Bin Center x", "@x"),
                    ("Bin Center y", "@y"),
                    ("Entries", "@h")]
        f.add_tools(bokeh.models.LassoSelectTool(select_every_mousemove=False),
                    bokeh.models.HoverTool(tooltips=tool_tip))

        s1 = bokeh.plotting.ColumnDataSource(data=dict(x=x_pos,
                                                       y=y_pos,
                                                       h=hist.flatten(),
                                                       color=colors)
                                             )
        f.square(source=s1,
                 size=self.size / bins,
                 color='color',
                 nonselection_alpha=0.3)

        f.x_range.start = self.xedges[0]
        f.x_range.end = self.xedges[-1]
        f.y_range.start = self.yedges[0]
        f.y_range.end = self.yedges[-1]
        f.xaxis.axis_label = x_label
        f.yaxis.axis_label = y_label

        self.selection_index = None
        s1.selected.js_on_change('indices',
                                 bokeh.models.CustomJS(args=dict(s1=s1), code=f"""
                var inds = cb_obj.indices;
                var kernel = IPython.notebook.kernel;
                IPython.notebook.kernel.execute("{self.name}.selection_index = " + inds);
            """)
                                 )
        return f

    def get_back_selected_items(self):
        if not self.selection_index:
            raise ValueError('No data selection found. Have you selected any data? '
                             'If yes you most likely have not intialized the DataSelctor correctly. '
                             'You have to callit as: my_instance_name = DataSelectionHist("my_instance_name")')
        m = np.isin(self.hist_inds, self.selection_index)
        return self.items[m]

    @staticmethod
    @numba.njit
    def _hist2d_with_index(xdata, ydata, weights, x_edges, y_edges):

        n_x_bins = len(x_edges) - 1
        n_y_bins = len(y_edges) - 1
        res_hist_inds = np.zeros(len(xdata), dtype=np.int32)
        res_hist = np.zeros((n_x_bins, n_y_bins), dtype=np.int64)

        # Create bin ranges:
        offset = 0
        for ind, xv in enumerate(xdata):
            yv = ydata[ind]
            w = weights[ind]
            hist_ind = 0
            found = False
            for ind_xb, low_xb in enumerate(x_edges[:-1]):
                high_xb = x_edges[ind_xb + 1]

                if not low_xb <= xv:
                    hist_ind += n_y_bins
                    continue
                if not xv < high_xb:
                    hist_ind += n_y_bins
                    continue

                # Checked both bins value is in bin, so check y:
                for ind_yb, low_yb in enumerate(y_edges[:-1]):
                    high_yb = y_edges[ind_yb + 1]

                    if not low_yb <= yv:
                        hist_ind += 1
                        continue
                    if not yv < high_yb:
                        hist_ind += 1
                        continue

                    found = True
                    res_hist_inds[offset] = hist_ind
                    res_hist[ind_xb, ind_yb] += w
                    offset += 1

            # Set to -1 if not in any
            if not found:
                res_hist_inds[offset] = -1
                offset += 1
        return res_hist, res_hist_inds

    def _make_bin_positions(self, bins, bin_range):
        """
        Helper function to create center positions for "histogram"
        markers.
        """
        edges = []
        for b, br in zip(bins, bin_range):
            # Create x and y edges
            d_range = br[1] - br[0]
            edges.append(np.arange(br[0], br[1] + d_range / b, d_range / b))

        # Convert into marker positions:
        xedges = edges[0]
        yedges = edges[1]
        self.xedges = xedges
        self.yedges = yedges
        x_pos = xedges[:-1] + np.diff(xedges) / 2
        x_pos = np.repeat(x_pos, len(yedges) - 1)

        y_pos = yedges[:-1] + np.diff(yedges) / 2
        y_pos = np.array(list(y_pos) * (len(xedges) - 1))
        return x_pos, y_pos

    def _get_color(self,
                   hist,
                   cmap_steps,
                   log_color_scale=False,
                   clim=(None, None),
                   undeflow_color=None,
                   overflow_color=None):
        """
        Helper function to create colorscale.
        """
        hist = hist.flatten()

        if clim[0] and undeflow_color:
            # If underflow is specified get indicies for underflow bins
            inds_underflow = np.argwhere(hist < clim[0]).flatten()

        if clim[1] and overflow_color:
            inds_overflow = np.argwhere(hist > clim[1]).flatten()

        # Clip data according to clim
        if np.any(clim):
            hist = np.clip(hist, clim[0], clim[1])

        self.clim = (np.min(hist), np.max(hist))
        if log_color_scale:
            color = np.log10(hist)
            color /= np.max(color)
            color *= cmap_steps - 1
        else:
            color = hist / np.max(hist)
            color *= cmap_steps - 1

        cmap = np.array(bokeh.palettes.viridis(cmap_steps))
        cmap = cmap[np.round(color).astype(np.int8)]

        if undeflow_color:
            cmap[inds_underflow] = undeflow_color

        if overflow_color:
            cmap[inds_overflow] = overflow_color
        return cmap
