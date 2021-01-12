import bokeh
import bokeh.plotting as bklt
import numpy as np
import strax
import straxen

# Default legend and color options for unknow, S1 and S2
COLORS = ('gray', 'blue', 'green')
LEGENDS = ('Unknown', 'S1', 'S2')


@straxen.mini_analysis(requires=('events', 'event_basics'))
def event_display_interactive(events, to_pe, run_id, context):
    """
    Interactive event display for XENONnT. Plots detailed main/alt
    S1/S2, bottom and top PMT hit pattern as well as all other peaks
    in a given event.

    Note:
        How to use:

        > straxen.event_display(context,
        >                       run_id,
        >                       time_range=(event['time'],
        >                                   event['endtime'])
        >                      )

    Warning:
        Raises an error if the user queries a time range which contains
        more than a single event.

    :return: bokeh.plotting.figure instance.
    """
    if len(events) > 1:
        raise ValueError('The time range you specified contains more'
                         ' than a single event. The event display only'
                         ' works with individual events for now.')
    peaks = context.get_array(run_id,
                         ('peaks', 'peak_basics', 'peak_positions'),
                         time_range=(events[0]['time'], events[0]['endtime']))

    # Select main/alt S1/S2s based on time and endtime in event:
    m_other_peaks = np.ones(len(peaks), dtype=np.bool_)  # To select non-event peaks
    endtime = strax.endtime(peaks)

    signal = {}
    keys = ['s1', 'alt_s1', 's2', 'alt_s2']
    for s in keys:
        m = (peaks['time'] == events[f'{s}_time']) & (endtime == events[f'{s}_endtime'])
        signal[s] = peaks[m]
        m_other_peaks &= ~m

    mes = 'This is odd was not able to find S1 and S2 for the specified event.'
    assert signal['s1'].shape[0] & signal['s2'].shape[0], mes
    print(signal['s1']['time'], signal['s1']['center_time'])
    # Use consistent labels:
    labels = {'s1': 'MS1', 'alt_s1': 'AS1', 's2': 'MS2', 'alt_s2': 'AS2'}

    # Detail plot S1/S2:
    fig_s1, _ = plot_peak_detail(signal['s1'], label=labels['s1'])
    if signal['alt_s1'].shape[0]:
        fig_s1, ps1 = plot_peak_detail(signal['alt_s1'], label=labels['alt_s1'], fig=fig_s1)
        ps1.visible = False

    fig_s2, _ = plot_peak_detail(signal['s2'], time_scaler=1000, label=labels['s2'])
    if signal['alt_s2'].shape[0]:
        fig_s2, ps2 = plot_peak_detail(signal['alt_s2'], time_scaler=1000, label=labels['alt_s2'], fig=fig_s2)
        ps2.visible = False

    # PMT arrays:
    # TOP
    for ind, k in enumerate(keys[2:] + keys[:2]):
        if not signal[k].shape[0]:
            # alt S1/S2 does not exist
            continue

        if not ind:
            # Main S2:
            fig_top, p, mapper = plot_pmt_array(signal[k][0], 'top', to_pe, label=labels[k])
            fig_top, p = plot_posS2s(signal[k][0], label=labels[k], fig=fig_top, s2_type=0)
        else:
            # Alt S2 and main/alt S1
            fig_top, p, _ = plot_pmt_array(signal[k][0], 'top', to_pe, label=labels[k], fig=fig_top)
            p.visible = False

    # Add also position for alt S2 and other S2s:
    fig_top, p = plot_posS2s(signal['alt_s2'][0], label=labels['alt_s2'], fig=fig_top, s2_type=1)
    p.visible = False
    m_other_s2 = m_other_peaks & (peaks['type'] == 2)
    fig_top, p = plot_posS2s(peaks[m_other_s2], label='OS2s', fig=fig_top, s2_type=2)
    p.visible = False

    # Same for bottom array, but without S2 pos:
    for ind, k in enumerate(keys):
        if not signal[k].shape[0]:
            # alt S1/S2 does not exist
            continue

        if not ind:
            # Main S1:
            fig_bottom, p, mapper = plot_pmt_array(signal[k][0], 'bottom', to_pe, label=labels[k])
        else:
            # Alt S1 and main/alt S2
            fig_bottom, p, _ = plot_pmt_array(signal[k][0], 'bottom', to_pe, label=labels[k], fig=fig_bottom)
            p.visible = False

    # Main Plot:
    title = _make_event_title(events[0], run_id)
    waveform = plot_peaks(peaks, time_scaler=1000)
    # Hightlight main and alternate S1/S2:
    start = peaks[0]['time']
    for s, p in signal.items():
        if p.shape[0]:
            pos = (p[0]['center_time'] - start) / 1000
            main = bokeh.models.Span(location=pos,
                                     dimension='height',
                                     line_alpha=0.6,
                                     )
            vline_label = bokeh.models.Label(x=pos,
                                             y=300,
                                             y_units='screen',
                                             angle=np.pi / 2,
                                             text=labels[s],
                                             )
            if 'alt' in s:
                main.line_dash = 'dotted'
            else:
                main.line_dash = 'dashed'
            waveform.add_layout(main)
            waveform.add_layout(vline_label)

    # Put everything together:
    upper_row = bokeh.layouts.Row(children=[fig_s1, fig_s2, fig_top, fig_bottom])

    event_display = bokeh.layouts.gridplot(children=[title, upper_row, waveform],
                                           sizing_mode='scale_both',
                                           ncols=1,
                                           merge_tools=True,
                                           toolbar_location='right',
                                           )

    return event_display


def plot_peak_detail(peak, time_scaler=1, label='', fig=None):
    """
    Function which makes a detailed plot for the given peak. As in the
    main/alt S1/S2 plots of the event display.

    :param peak: Peak to be plotted.
    :param time_scaler: Factor to rescale the time from ns to other scale.
        E.g. =1000 scales to µs.
    :param label: Label to be used in the plot legend.
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
                                                  time_scaler=time_scaler,
                                                  keep_amplitude_per_sample=False
                                                  )

    patches = fig.patches(source=source,
                          legend_label=label,
                          fill_color=COLORS[p_type],
                          fill_alpha=0.2,
                          line_color=COLORS[p_type],
                          line_width=0.5,
                          name=label
                          )
    fig.xaxis.axis_label = f"Time [{time_scaler} ns]"
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


def plot_peaks(peaks, time_scaler=1, fig=None):
    """
    Function which plots a list/array of peaks relative to the first
    one.

    :param peaks: Peaks to be plotted.
    :param time_scaler:  Factor to rescale the time from ns to other scale.
        E.g. =1000 scales to µs.
    :param fig: Instance of bokeh.plotting.figure if None one will be
        created via straxen.bokeh.utils.default_figure().
    :return: bokeh.plotting.figure instance.
    """
    if not fig:
        fig = straxen.bokeh_utils.default_fig(width=1600, height=400)

    for i in range(0, 3):
        _ind = np.where(peaks['type'] == i)[0]

        source = straxen.bokeh_utils.get_peaks_source(peaks[_ind],
                                                      relative_start=peaks[0]['time'],
                                                      time_scaler=time_scaler,
                                                      )
        if i == 1:
            print(source.data['time'], source.data['center_time'])
        fig.patches(source=source,
                    fill_color=COLORS[i],
                    fill_alpha=0.2,
                    line_color=COLORS[i],
                    line_width=0.5,
                    legend_label=LEGENDS[i],
                    name=LEGENDS[i],
                    )

        tt = straxen.bokeh_utils.peak_tool_tip(i)
        tt = [v for k, v in tt.items() if k != 'time_dynamic']
        fig.add_tools(bokeh.models.HoverTool(names=[LEGENDS[i]], tooltips=tt))

    fig.xaxis.axis_label = 'Time [µs]'
    fig.xaxis.axis_label_text_font_size = '14pt'
    fig.yaxis.axis_label = "Amplitude [pe/sample]"
    fig.yaxis.axis_label_text_font_size = '14pt'

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    return fig


def plot_pmt_array(peak, array_type, to_pe, fig=None, label=''):
    """
    Plots top or bottom PMT array for given peak.

    :param peak: Peak for which the hit pattern should be plotted.
    :param array_type: String which specifies if "top" or "bottom"
        PMT array should be plotted
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
                ('Y-Postion [cn]', '$y'),
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
    pmts = straxen.pmt_positions()
    m = to_pe == 0
    pmts_on = pmts[~m]
    pmts_on = pmts_on[pmts_on['array'] == array_type]

    if np.any(m):
        pmts_off = pmts[m]
        pmts_off = pmts_off[pmts_off['array'] == array_type]
        fig = _plot_off_pmts(pmts_off, fig)

    area_per_channel = peak['area_per_channel'][pmts_on['i']]

    mapper = bokeh.transform.linear_cmap(field_name='area',
                                         palette="Viridis256",
                                         low=min(area_per_channel),
                                         high=max(area_per_channel))

    source_on = bklt.ColumnDataSource(data={'x': pmts_on['x'],
                                            'y': pmts_on['y'],
                                            'area': area_per_channel,
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


def plot_posS2s(peaks, label='', fig=None, s2_type=0):
    """
    Plots xy-positions of specified peaks.

    :param peaks: Peaks for which the position should be plotted.
    :param label: Legend label and plot name (name serves as idenitfier).
    :param fig: bokeh.plotting.figure instance the plot should be plotted
        into. If None creates new instance.
    :param s2_type: 0 plots main S2 style, 1 for alt S2 and
        2 for other S2s (e.g. single electrons).
    """
    if not np.all(peaks['type'] == 2):
        raise ValueError('All peaks must be S2!')

    if not fig:
        fig = straxen.bokeh_utils.default_fig()

    source = straxen.bokeh_utils.get_peaks_source(peaks)

    if s2_type == 0:
        p = fig.cross(source=source,
                      name=label,
                      legend_label=label,
                      color='red',
                      line_width=2,
                      size=12)

    if s2_type == 1:
        p = fig.cross(source=source,
                      name=label,
                      legend_label=label,
                      color='orange',
                      angle=45 / 360 * 2 * np.pi,
                      line_width=2,
                      size=12)

    if s2_type == 2:
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


def _make_event_title(event, run_id, width='1600px'):
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
                             style={'text-align': 'center',
                                    'width': width,
                                    },
                             sizing_mode='scale_both',
                             )
    return title
