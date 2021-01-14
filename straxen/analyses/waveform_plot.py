import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
import strax
import straxen
from mpl_toolkits.axes_grid1 import inset_locator
from datetime import datetime
from .records_matrix import DEFAULT_MAX_SAMPLES

export, __all__ = strax.exporter()
__all__ += ['plot_wf']


@straxen.mini_analysis()
def plot_waveform(context,
                  deep=False,
                  show_largest=100,
                  figsize=None,
                  max_samples=DEFAULT_MAX_SAMPLES,
                  ignore_max_sample_warning=True,
                  cbar_loc='lower right',
                  lower_panel_height=2,
                  **kwargs):
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
                                    max_samples=max_samples,
                                    ignore_max_sample_warning=ignore_max_sample_warning,
                                    raw=deep == 'raw',
                                    single_figure=False)

        straxen.quiet_tight_layout()
        plt.subplots_adjust(hspace=0)


@straxen.mini_analysis(
    requires=('peaks', 'peak_basics'),
    default_time_selection='touching',
    warn_beyond_sec=60)
def plot_peaks(peaks,
               seconds_range,
               t_reference,
               show_largest=100,
               single_figure=True,
               figsize=(10, 4),
               xaxis=True):
    if single_figure:
        plt.figure(figsize=figsize)
    plt.axhline(0, c='k', alpha=0.2)

    peaks = peaks[np.argsort(-peaks['area'])[:show_largest]]
    peaks = strax.sort_by_time(peaks)

    for p in peaks:
        plot_peak(p,
                  t0=t_reference,
                  color={0: 'gray', 1: 'b', 2: 'g'}[p['type']])

    if xaxis == 'since_start':
        seconds_range_xaxis(seconds_range, t0=seconds_range[0])
    elif xaxis:
        seconds_range_xaxis(seconds_range)
        plt.xlim(*seconds_range)
    else:
        plt.xticks([])
        plt.xlim(*seconds_range)
    plt.ylabel("Intensity [PE/ns]")
    if single_figure:
        plt.tight_layout()


@straxen.mini_analysis(
    requires=('peaks', 'peak_basics'),
    default_time_selection='touching',
    warn_beyond_sec=60)
def plot_hit_pattern(peaks,
                     seconds_range,
                     t_reference,
                     axes=None,
                     vmin=None,
                     log_scale=False,
                     label=None,
                     single_figure=False,
                     xenon1t=False,
                     figsize=(10, 4), ):
    if single_figure:
        plt.figure(figsize=figsize)
    if len(peaks) > 1:
        print(f'warning showing total area of {len(peaks)} peaks')
    straxen.plot_pmts(np.sum(peaks['area_per_channel'], axis=0),
                      axes=axes, vmin=vmin, log_scale=log_scale, label=label,
                      xenon1t=xenon1t)


@straxen.mini_analysis()
def plot_records_matrix(context, run_id,
                        seconds_range,
                        cbar_loc='upper right',
                        raw=False,
                        single_figure=True, figsize=(10, 4),
                        max_samples=DEFAULT_MAX_SAMPLES,
                        ignore_max_sample_warning=False,
                        **kwargs):
    if seconds_range is None:
        raise ValueError(
            "You must pass a time selection (e.g. seconds_range) "
            "to plot_records_matrix.")

    if single_figure:
        plt.figure(figsize=figsize)

    f = context.raw_records_matrix if raw else context.records_matrix

    wvm, ts, ys = f(run_id,
                    max_samples=max_samples,
                    ignore_max_sample_warning=ignore_max_sample_warning,
                    **kwargs)

    plt.pcolormesh(
        ts, ys, wvm.T,
        norm=matplotlib.colors.LogNorm(),
        vmin=min(0.1 * wvm.max(), 1e-2),
        vmax=wvm.max(),
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


def seconds_range_xaxis(seconds_range, t0=None):
    """Make a pretty time axis given seconds_range"""
    plt.xlim(*seconds_range)
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    xticks = plt.xticks()[0]
    if not len(xticks):
        return

    # Format the labels
    # I am not very proud of this code...
    def chop(x):
        return np.floor(x).astype(np.int)

    if t0 is None:
        xticks_ns = np.round(xticks * int(1e9)).astype(np.int)
    else:
        xticks_ns = np.round((xticks - xticks[0]) * int(1e9)).astype(np.int)
    sec = chop(xticks_ns // int(1e9))
    ms = chop((xticks_ns % int(1e9)) // int(1e6))
    us = chop((xticks_ns % int(1e6)) // int(1e3))
    samples = chop((xticks_ns % int(1e3)) // 10)

    labels = [str(sec[i]) for i in range(len(xticks))]
    print_ns = np.any(samples != samples[0])
    print_us = print_ns | np.any(us != us[0])
    print_ms = print_us | np.any(ms != ms[0])
    if print_ms and t0 is None:
        labels = [l + f'.{ms[i]:03}' for i, l in enumerate(labels)]
        if print_us:
            labels = [l + r' $\bf{' + f'{us[i]:03}' + '}$'
                      for i, l in enumerate(labels)]
            if print_ns:
                labels = [l + f' {samples[i]:02}0' for i, l in enumerate(labels)]
        plt.xticks(ticks=xticks, labels=labels, rotation=90)
    else:
        labels = list(chop((xticks_ns // 10) * 10))
        labels[-1] = ""
        plt.xticks(ticks=xticks, labels=labels, rotation=0)
    if t0 is None:
        plt.xlabel("Time since run start [sec]")
    else:
        plt.xlabel("Time [ns]")


def plot_peak(p, t0=None, center_time=True, **kwargs):
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

    # Mark center time with thin black line
    if center_time:
        if t0 is None:
            t0 = p['time']
        ct = (p['center_time'] - t0) / int(1e9)
        plt.axvline(ct, c='k', alpha=0.4, linewidth=1, linestyle='--')


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


def plot_wf(st: strax.Context,
            containers,
            run_id, plot_log=True, plot_extension=0, hit_pattern=True,
            timestamp=True, time_fmt="%d-%b-%Y (%H:%M:%S)",
            **kwargs):
    """
    Combined waveform plot
    :param st: strax.Context
    :param containers: peaks/records/events where from we want to plot
        all the peaks that are within it's time range +- the
        plot_extension. For example, you can provide three adjacent
        peaks and plot them in a single figure.
    :param run_id: run_id of the containers
    :param plot_log: Plot the y-scale of the wf in log-space
    :param plot_extension: include this much nanoseconds around the
        containers (can be scalar or list of (-left_extension,
        right_extension).
    :param hit_pattern: include the hit-pattern in the wf
    :param timestamp: print the timestamp to the plot
    :param time_fmt: format fo the timestamp (datetime.strftime format)
    :param kwargs: kwargs for plot_peaks
    """

    if not isinstance(run_id, str):
        raise ValueError(f'Insert single run_id, not {run_id}')

    p = containers  # usually peaks
    run_start, _ = st.estimate_run_start_and_end(run_id)
    t_range = np.array([p['time'].min(), strax.endtime(p).max()])

    # Extend the time range if needed.
    if not np.iterable(plot_extension):
        t_range += np.array([-plot_extension, plot_extension])
    elif len(plot_extension) == 2:
        if not plot_extension[0] < 0:
            warnings.warn('Left extension is positive (i.e. later than start '
                          'of container).')
        t_range += plot_extension
    else:
        raise ValueError('Wrong dimensions for plot_extension. Use scalar or '
                         'object of len( ) == 2')
    t_range -= run_start
    t_range = t_range / 10 ** 9
    t_range = np.clip(t_range, 0, np.inf)

    if hit_pattern:
        plt.figure(figsize=(14, 11))
        plt.subplot(212)
    else:
        plt.figure(figsize=(14, 5))
    # Plot the wf
    plot_peaks(st, run_id, seconds_range=t_range, single_figure=False, **kwargs)

    if timestamp:
        _ax = plt.gca()
        t_stamp = datetime.datetime.fromtimestamp(
            containers['time'].min() / 10 ** 9).strftime(time_fmt)
        _ax.text(0.975, 0.925, t_stamp,
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=_ax.transAxes)
    # Select the additional two panels to show the top and bottom arrays
    if hit_pattern:
        axes = plt.subplot(221), plt.subplot(222)
        plot_hit_pattern(st, run_id,
                         seconds_range=t_range,
                         axes=axes,
                         vmin=1 if plot_log else None,
                         log_scale=plot_log,
                         label='Area per channel [PE]')
