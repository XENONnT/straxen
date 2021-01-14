import bokeh.plotting as bklt
import numpy as np
import strax

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

    x = [0] + list(np.arange(peak['length'])) + [peak['length']]
    x = np.array(x) * peak['dt'] + peak['time']
    y = [0] + list(peak['data'][:peak['length']]) + [0]
    y = np.array(y) / dt_a
    return x, y


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


def default_fig(width=400, height=400, title=''):
    """
    Helper function which returns a bokeh.plotting.figure instance
    with sizing_mode set to 'scale_both' and an aspect ratio set
    according to the specified width and height.

    :param width: Plot width in pixels
    :param height: PLot height in pixels.
    :param title: Title of the plot.

    :returns: bokeh.plotting.figure  instance.
    """
    fig = bklt.figure(plot_width=width,
                      plot_height=height,
                      sizing_mode='scale_both',
                      aspect_ratio=width / height,
                      title=title,
                      tools="pan,box_zoom,reset"
                      )
    return fig
