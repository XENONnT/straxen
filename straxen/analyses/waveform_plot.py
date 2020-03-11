import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import strax
import straxen
from mpl_toolkits.axes_grid1 import inset_locator


@straxen.mini_analysis()
def plot_waveform(context, deep=False, show_largest=100, figsize=None,
                  cbar_loc='lower right', lower_panel_height=2, **kwargs):
    """Plot the sum waveform and optionally per-PMT waveforms

    :param deep: If True, show per-PMT waveform matrix under sum waveform.
    If 'raw', use raw_records instead of records to do so.
    :param show_largest: Show only the largest show_largest peaks.
    :param figsize: Matplotlib figure size for the plot

    Additional options for deep = True or raw:
    :param cbar_loc: location of the intensity color bar. Set to None
    to omit it altogether.
    :param lower_panel_height: Height of the lower panel in terms of
    the height of the upper panel.
    """
    if figsize is None:
        figsize = (10, 6 if deep else 4)

    if not deep:
        context.plot_peaks(**kwargs, show_largest=show_largest, figsize=figsize)

    else:
        f, axes = plt.subplots(2, 1,
                               figsize=figsize,
                               gridspec_kw={'height_ratios': [1, lower_panel_height]})

        plt.sca(axes[0])
        context.plot_peaks(**kwargs, show_largest=show_largest,
                           single_figure=False,
                           xaxis=False)

        plt.sca(axes[1])
        context.plot_records_matrix(**kwargs,
                                    cbar_loc=cbar_loc,
                                    raw=deep == 'raw',
                                    single_figure=False)

        straxen.quiet_tight_layout()
        plt.subplots_adjust(hspace=0)


@straxen.mini_analysis(
    requires=('peaks', 'peak_basics'),
    default_time_selection='touching',
    warn_beyond_sec=60)
def plot_peaks(peaks, seconds_range, t_reference, show_largest=100,
               single_figure=True, figsize=(10, 4), xaxis=True):
    if single_figure:
        plt.figure(figsize=figsize)
    plt.axhline(0, c='k', alpha=0.2)

    peaks = peaks[np.argsort(-peaks['area'])[:show_largest]]
    peaks = strax.sort_by_time(peaks)

    for p in peaks:
        plot_peak(p,
                  t0=t_reference,
                  color={0: 'gray', 1: 'b', 2: 'g'}[p['type']])

    if xaxis:
        seconds_range_xaxis(seconds_range)
    else:
        plt.xticks([])
    plt.xlim(*seconds_range)
    plt.ylabel("Intensity [PE/ns]")
    if single_figure:
        plt.tight_layout()


@straxen.mini_analysis()
def plot_records_matrix(context, run_id,
                        seconds_range,
                        cbar_loc='upper right',
                        raw=False,
                        single_figure=True, figsize=(10, 4),
                        **kwargs):
    if seconds_range is None:
        raise ValueError(
            "You must pass a time selection (e.g. seconds_range) "
            "to plot_records_matrix.")

    if single_figure:
        plt.figure(figsize=figsize)

    f = context.raw_records_matrix if raw else context.records_matrix

    wvm, ts, ys = f(run_id, **kwargs)

    plt.pcolormesh(
        ts, ys, wvm.T,
        norm=matplotlib.colors.LogNorm(), vmin=1e-2,
        cmap=plt.cm.inferno)
    plt.xlim(*seconds_range)

    ax = plt.gca()
    seconds_range_xaxis(seconds_range)
    ax.invert_yaxis()
    plt.ylabel("PMT Number")

    if cbar_loc is not None:
        # Create a white box to place the color bar in
        # See https://stackoverflow.com/questions/18211967
        bbox = inset_locator.inset_axes(ax,
                                        width="20%", height="22%",
                                        loc=cbar_loc)
        [bbox.spines[k].set_visible(False) for k in bbox.spines]
        bbox.patch.set_facecolor((1, 1, 1, 0.9))
        bbox.set_xticks([])
        bbox.set_yticks([])

        # Create the actual color bar
        cax = inset_locator.inset_axes(bbox,
                                       width='90%', height='20%',
                                       loc='upper center')
        plt.colorbar(cax=cax,
                     label='Intensity [PE/ns]',
                     orientation='horizontal')
        cax.xaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%g'))

    plt.sca(ax)

    if single_figure:
        straxen.quiet_tight_layout()


def seconds_range_xaxis(seconds_range):
    """Make a pretty time axis given seconds_range"""
    plt.xlim(*seconds_range)
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    xticks = plt.xticks()[0]
    if not len(xticks):
        return
    #xticks[0] = seconds_range[0]
    #xticks[-1] = seconds_range[-1]

    # Format the labels
    # I am not very proud of this code...
    def chop(x):
        return np.floor(x).astype(np.int)
    xticks_ns = np.round(xticks * int(1e9)).astype(np.int)
    sec = chop(xticks_ns // int(1e9))
    ms = chop((xticks_ns % int(1e9)) // int(1e6))
    us = chop((xticks_ns % int(1e6)) // int(1e3))
    samples = chop((xticks_ns % int(1e3)) // 10)

    labels = [str(sec[i]) for i in range(len(xticks))]
    print_samples = np.any(samples != samples[0])
    print_us = print_samples | np.any(us != us[0])
    print_ms = print_us | np.any(ms != ms[0])
    if print_ms:
        labels = [l + f'.{ms[i]:03}' for i, l in enumerate(labels)]
        if print_us:
            labels = [l + r' $\bf{' + f'{us[i]:03}' + '}$'
                      for i, l in enumerate(labels)]
            if print_samples:
                labels = [l + f' {samples[i]:02}' for i, l in enumerate(labels)]

    plt.xticks(ticks=xticks, labels=labels, rotation=90)
    plt.xlim(*seconds_range)
    plt.xlabel("Time since run start [sec]")


def plot_peak(p, t0=None, **kwargs):
    x, y = time_and_samples(p, t0=t0)
    kwargs.setdefault('linewidth', 1)

    # Plot waveform
    plt.plot(x, y,
             drawstyle='steps-pre',
             **kwargs)
    if 'linewidth' in kwargs:
        del kwargs['linewidth']
    kwargs['alpha'] = kwargs.get('alpha', 1) * 0.2
    plt.fill_between(x, 0, y, step='pre', linewidth=0, **kwargs)

    # Mark extent with thin black line
    plt.plot([x[0], x[-1]], [y.max(), y.max()],
             c='k', alpha=0.3, linewidth=1)


def time_and_samples(p, t0=None):
    """Return (x, y) numpy arrays for plotting the waveform data in p
    using 'steps-pre'.

    Where x is the time since t0 in seconds (or another time_scale),
      and y is intensity in PE / ns.
    :param p: Peak or other similar strax data type
    :param t0: Zero of time in ns since unix epoch
    """
    n = p['length']
    if t0 is None:
        t0 = p['time']
    x = ((p['time'] - t0) + np.arange(n + 1) * p['dt']) / int(1e9)
    y = p['data'][:n] / p['dt']
    return x, np.concatenate([[y[0]], y])
