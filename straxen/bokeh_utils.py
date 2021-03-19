import bokeh.plotting as bklt
import bokeh
import numpy as np
import strax

export, __all__ = strax.exporter()


@export
def bokeh_to_wiki(fig, outputfile=None):
    """
    Function which converts bokeh HTML code to a wiki readable code.

    :param fig: Figure to be conerted
    :param outputfile: String of absolute file path. If specified output
        is writen to the file. Else output is print to the notebook and
        can be simply copied into the wiki.
    """
    # convert plot to wiki format:
    html = bokeh.embed.file_html(fig, bokeh.resources.CDN)
    html = '\n'.join((['<html>'] + html.split('\n')[6:]))

    if outputfile:
        with open(outputfile, mode='w') as file:
            file.write(html)
    else:
        print(html)


def get_peaks_source(peaks, relative_start=0, time_scaler=1, keep_amplitude_per_sample=True):
    """
    Computes bokeh.plotting.ColumnDataSource for given peaks.

    :param peaks: Peaks to be plotted.
    :param relative_start: t0 from which on the peaks should be plotted.
    :param time_scaler: Factor to rescale the time from ns to other scale.
        E.g. =1000 scales to µs.
    :param keep_amplitude_per_sample: Boolean if true amplitude of the
        plotted peaks is in pe/sample. False pe/ns.
    :return: bokeh.plotting.ColumnDataSource instance which can be used
        to plot peaks.
    """
    if not (np.all(peaks['type'] == peaks[0]['type'])):
        raise ValueError('All peaks must be of the same type (S1, S2 or Unknown)!')

    x_p = []
    y_p = []
    for ind, p in enumerate(peaks):
        x, y = _patches_x_y(p, keep_amplitude_per_sample=keep_amplitude_per_sample)
        x -= relative_start  # relative to first peak
        x = x / time_scaler
        x_p.append(x)
        y_p.append(y)

    if peaks[0]['type'] == 2:
        scaler = 10**-3
    else:
        scaler = 1

    source = bklt.ColumnDataSource(data={'xs': x_p,  # Coordinates for peak patches
                                         'ys': y_p,
                                         'x': peaks['x'],  # XY-Pos in PMZ Hitpattern
                                         'y': peaks['y'],
                                         'dt': peaks['dt'],
                                         'time': peaks['time'],
                                         'center_time': peaks['center_time'],
                                         'endtime': strax.endtime(peaks),
                                         'width_50': peaks['range_50p_area'] * scaler,
                                         'width_90': peaks['range_90p_area'] * scaler,
                                         'rise': peaks['rise_time'] * scaler,
                                         'rel_center_time': (peaks['center_time'] - peaks['time']) * scaler,
                                         'area': peaks['area'],
                                         'aft': peaks['area_fraction_top'],
                                         'nhits': peaks['n_hits'],
                                         }
                                   )
    return source


def _patches_x_y(peak, keep_amplitude_per_sample=False):
    """
    Creates x,y coordinates needed to draw peaks via
    bokeh.models.patches.

    :param peak: Peak for which we need the x/y samples
    :param keep_amplitude_per_sample: If y-data should be in units of "per sample"
        or "per ns".

    :returns: Tuple of x, y
    """
    if keep_amplitude_per_sample:
        dt_a = 1
    else:
        dt_a = peak['dt']

    x = np.arange(peak["length"])
    xx = np.zeros(2 * len(x), dtype=x.dtype)
    mx = 0.5 * (x[1::] + x[:-1])
    xx[1:-1:2] = mx
    xx[2::2] = mx
    xx[0] = 1.5 * x[0] - 0.5 * x[1]
    xx[-1] = 1.5 * x[-1] - 0.5 * x[-2]
    xx = np.array(xx) * peak['dt'] + peak['time']
    y = peak['data'][:peak['length']]
    yy = np.zeros(2 * len(x))
    yy[0::2] = y
    yy[1::2] = y
    yy = np.array(yy) / dt_a

    # baseline since we'll fill underneath
    xx = np.concatenate([[xx[0]], xx, [xx[-1]]])
    yy = np.concatenate([[0], yy, [0]])
    return xx, yy


def peak_tool_tip(peak_type):
    """
    Default mouseover tooltip for peaks.

    :param peak_type: If 2, all time variables are in µs else in ns.
    :return: dictionary of tooltips. Can be converted to a list for
        bokeh.models.HoverTool.
    """

    # Add static time parameters:
    tool_tip = {"time_static": ("time [ns]", "@time"),
                "center_time": ("center_time [ns]", "@center_time"),
                "endtime": ("endtime [ns]", "@endtime"),
                }

    # Now ns/µs parameters for S1 and S2
    tool_tip['dt'] = ("dt [ns/sample]", "@dt")
    tool_tip['time_dynamic'] = ("time [ns]", "$x")
    tool_tip['rel_center_time'] = ('center time [ns]', '@rel_center_time')
    tool_tip['range_50p_width'] = ('50% width [ns]', '@width_50')
    tool_tip['range_90p_width'] = ('90% width [ns]', '@width_90')
    tool_tip['rise_time'] = ('rise time [ns]', '@rise')

    # Add non-time parameters (results in an ordered tooltip)
    tool_tip['amplitude'] = ("Amplitude [pe/ns]", "$y")
    tool_tip["area"] = ('area [pe]', '@area')
    tool_tip["aft"] = ('AFT', '@aft')
    tool_tip["nhits"] = ('nhits', '@nhits')

    if peak_type == 2:
        for k, i in tool_tip.items():
            if k not in ["time_static", "center_time", "endtime"]:
                tool_tip[k] = (i[0].replace('[ns]', '[µs]'), i[1])

    return tool_tip


def default_fig(width=400, height=400, title='', **kwargs):
    """
    Helper function which returns a bokeh.plotting.figure instance
    with sizing_mode set to 'scale_both' and an aspect ratio set
    according to the specified width and height.

    :param width: Plot width in pixels
    :param height: PLot height in pixels.
    :param title: Title of the plot.

    Also allows for additional kwargs accepted by bokeh.plotting.

    :returns: bokeh.plotting.figure  instance.
    """
    fig = bklt.figure(plot_width=width,
                      plot_height=height,
                      sizing_mode='scale_both',
                      aspect_ratio=width / height,
                      title=title,
                      tools="pan,box_zoom,reset,save",
                      **kwargs
                      )
    return fig
