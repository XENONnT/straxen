import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import strax
import straxen
from mpl_toolkits.axes_grid1 import inset_locator
from datetime import datetime
import pytz
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


# Default attributes to display in the event_display (looks little
# complicated but just repeats same fields for S1 S1)
# Should be of form as below where {v} wil be filled with the value of
# event['key']:
#  (('key', '{v} UNIT'), ..)
PEAK_DISPLAY_DEFAULT_INFO = sum([[(k.format(i=s_i), u) for k, u in
                                  (('cs{i}', '{v:.2f} PE'),
                                   ('s{i}_area', '{v:.2f} PE'),
                                   ('alt_cs{i}', '{v:.2f} PE'),
                                   ('s{i}_n_channels', '{v}'),
                                   ('s{i}_area_fraction_top', '{v:.2f}'),
                                   ('s{i}_range_50p_area', '{v:.1f}'),
                                   )] for s_i in (1, 2)], [])
EVENT_DISPLAY_DEFAULT_INFO = (('time', '{v} ns'),
                              ('endtime', '{v} ns'),
                              ('event_number', '{v}'),
                              ('x', '{v:.2f} cm'),
                              ('y', '{v:.2f} cm'),
                              ('z', '{v:.2f} cm'),
                              ('r', '{v:.2f} cm'),
                              ('theta', '{v:.2f} rad'),
                              ('drift_time', '{v} ns'),
                              ('alt_s1_interaction_drift_time', '{v} ns'),
                              ('alt_s2_interaction_drift_time', '{v} ns')
                              )


@straxen.mini_analysis(requires=('event_info',))
def event_display(context,
                  run_id,
                  events,
                  records_matrix=True,
                  s2_fuzz=50,
                  s1_fuzz=0,
                  max_peaks=500,
                  xenon1t=False,
                  display_peak_info=PEAK_DISPLAY_DEFAULT_INFO,
                  display_event_info=EVENT_DISPLAY_DEFAULT_INFO,
                  s1_hp_kwargs=None,
                  s2_hp_kwargs=None,
                  ):
    """
    Make a waveform-display of a given event. Requires events, peaks and
        peaklets (optionally: records). NB: time selection should return
        only one event!

    :param context: strax.Context provided by the minianalysis wrapper
    :param run_id: run-id of the event
    :param events: events, provided by the minianalysis wrapper
    :param records_matrix: False (no record matrix), True, or "raw"
        (show raw-record matrix)
    :param s2_fuzz: extra time around main S2 [ns]
    :param s1_fuzz: extra time around main S1 [ns]
    :param max_peaks: max peaks for plotting in the wf plot
    :param xenon1t: True: is 1T, False: is nT
    :param display_peak_info: tuple, items that will be extracted from
        event and displayed in the event info panel see above for format
    :param display_event_info: tuple, items that will be extracted from
        event and displayed in the peak info panel see above for format
    :param s1_hp_kwargs: dict, optional kwargs for S1 hitpatterns
    :param s2_hp_kwargs: dict, optional kwargs for S2 hitpatterns
    :return: axes used for plotting:
        ax_s1, ax_s2, ax_s1_hp_t, ax_s1_hp_b,
        ax_event_info, ax_peak_info, ax_s2_hp_t, ax_s2_hp_b,
        ax_ev,
        ax_rec
        Where those panels (axes) are:
            - ax_s1, main S1 peak
            - ax_s2, main S2 peak
            - ax_s1_hp_t, S1 top hit pattern
            - ax_s1_hp_b, S1 bottom hit pattern
            - ax_s2_hp_t, S2 top hit pattern
            - ax_s2_hp_b, S2 bottom hit pattern
            - ax_event_info, text info on the event
            - ax_peak_info, text info on the main S1 and S2
            - ax_ev, waveform of the entire event
            - ax_rec, (raw)record matrix (if any otherwise None)
    """
    if len(events) != 1:
        raise ValueError(f'Found {len(events)} only request one')
    event = events[0]
    if records_matrix not in ('raw', True, False):
        raise ValueError('Choose either "raw", True or False for records_matrix')
    if ((records_matrix == 'raw' and not context.is_stored(run_id, 'raw_records')) or
            (records_matrix and not context.is_stored(run_id, 'records'))):
        print("(raw)records not stored! Not showing records_matrix")
        records_matrix = False

    fig = plt.figure(figsize=(25, 7 * (2 + int(records_matrix))),
                     facecolor='white')
    grid = plt.GridSpec((2 + int(records_matrix)), 1, hspace=0.1,
                        height_ratios=[1.5, 0.5, 0.5][:2 + int(records_matrix)]
                        )

    # S1, S2, hitpatterns
    gss_0 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=grid[0], wspace=0.25, hspace=0.4)
    ax_s1 = fig.add_subplot(gss_0[0])
    ax_s2 = fig.add_subplot(gss_0[1])
    ax_s1_hp_t = fig.add_subplot(gss_0[2])
    ax_s1_hp_b = fig.add_subplot(gss_0[3])
    ax_s2_hp_t = fig.add_subplot(gss_0[6])
    ax_s2_hp_b = fig.add_subplot(gss_0[7])

    # Peak & event info
    ax_event_info = fig.add_subplot(gss_0[4])
    ax_peak_info = fig.add_subplot(gss_0[5])

    # All peaks in event
    gss_1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid[1])
    ax_ev = fig.add_subplot(gss_1[0])

    # titles
    ax_s1.set_title('Main S1')
    ax_s1_hp_t.set_title('S1 top')
    ax_s1_hp_b.set_title('S1 bottom')
    ax_s2.set_title('Main S2')
    ax_s2_hp_t.set_title('S2 top')
    ax_s2_hp_b.set_title('S2 bottom')
    ax_event_info.set_title('Event info')
    ax_peak_info.set_title('Peak info')
    ax_rec = None

    # (raw)records matrix (optional)
    if records_matrix:
        gss_2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid[2])
        ax_rec = fig.add_subplot(gss_2[0])

    # Parse the hit pattern options
    # Convert to dict (not at function definition because of mutable defaults)
    if s1_hp_kwargs is None:
        s1_hp_kwargs = {}
    if s2_hp_kwargs is None:
        s2_hp_kwargs = {}
    # Hit patterns options:
    for hp_opt, color_map in ((s1_hp_kwargs, "Blues"), (s2_hp_kwargs, "Greens")):
        _common_opt = dict(xenon1t=xenon1t,
                           pmt_label_color='lightgrey',
                           log_scale=True,
                           vmin=0.1,
                           s=(250 if records_matrix else 220),
                           pmt_label_size=7,
                           edgecolor='grey',
                           cmap=color_map)
        # Is there a better way to do this update?
        hp_opt.update(_common_opt)
        _update = hp_opt.copy()
        hp_opt.update(_update)

    # S1
    if events['s1_area'] > 0:
        plt.sca(ax_s1)
        context.plot_peaks(run_id,
                           time_range=(events['s1_time'] - s1_fuzz,
                                       events['s1_endtime'] + s1_fuzz),
                           single_figure=False)

        # Hit pattern plots
        area = context.get_array(run_id, 'peaklets',
                                 time_range=(events['s1_time'],
                                             events['s1_endtime']),
                                 keep_columns=('area_per_channel', 'time', 'dt', 'length'))
        for ax, array in ((ax_s1_hp_t, 'top'), (ax_s1_hp_b, 'bottom')):
            plt.sca(ax)
            straxen.plot_on_single_pmt_array(c=np.sum(area['area_per_channel'], axis=0),
                                             array_name=array,
                                             **s1_hp_kwargs)
            # Mark reconstructed position
            plt.scatter(event['x'], event['y'], marker='X', s=100, c='r')

    # S2
    if events['s2_area'] > 0:
        plt.sca(ax_s2)
        context.plot_peaks(run_id,
                           time_range=(events['s2_time'] - s2_fuzz,
                                       events['s2_endtime'] + s2_fuzz),
                           single_figure=False)

        # Hit pattern plots
        area = context.get_array(run_id, 'peaklets',
                                 time_range=(events['s2_time'],
                                             events['s2_endtime']),
                                 keep_columns=('area_per_channel', 'time', 'dt', 'length'))
        for ax, array in ((ax_s2_hp_t, 'top'), (ax_s2_hp_b, 'bottom')):
            plt.sca(ax)
            straxen.plot_on_single_pmt_array(c=np.sum(area['area_per_channel'], axis=0),
                                             array_name=array,
                                             **s2_hp_kwargs)
            # Mark reconstructed position
            plt.scatter(event['x'], event['y'], marker='X', s=100, c='r')

    # Fill panels with peak/event info
    for it, (ax, labels_and_unit) in enumerate([(ax_event_info, display_event_info),
                                                (ax_peak_info, display_peak_info)]):
        for i, (_lab, _unit) in enumerate(labels_and_unit):
            coord = 0.01, 0.9 - 0.9 * i / len(labels_and_unit)
            ax.text(*coord, _lab, va='top', zorder=-10)
            ax.text(coord[0] + 0.5, coord[1],
                    _unit.format(v=event[_lab]), va='top', zorder=-10)
            # Remove axes and labels from panel
            ax.set_xticks([])
            ax.set_yticks([])
            _ = [s.set_visible(False) for s in ax.spines.values()]

    # Plot peaks in event
    plt.sca(ax_ev)
    context.plot_peaks(run_id,
                       time_range=(events['time'], events['endtime']),
                       show_largest=max_peaks,
                       single_figure=False)
    ev_range = plt.xlim()

    if records_matrix:
        plt.sca(ax_rec)
        context.plot_records_matrix(run_id,
                                    raw=records_matrix == 'raw',
                                    time_range=(events['time'],
                                                events['endtime']),
                                    single_figure=False)
        ax_rec.tick_params(axis='x', rotation=0)
        if not xenon1t:
            # Top vs bottom division
            ax_rec.axhline(straxen.n_top_pmts, c='k')
        plt.xlim(*ev_range)

    # Final tweaks
    ax_s2.tick_params(axis='x', rotation=45)
    ax_s1.tick_params(axis='x', rotation=45)
    ax_ev.tick_params(axis='x', rotation=0)
    title = (f'Run {run_id}. Time '
             f'{str(events["time"])[:-9]}.{str(events["time"])[-9:]}\n'
             f'{datetime.fromtimestamp(event["time"]/1e9, tz=pytz.utc)}')
    plt.suptitle(title, y=0.95)
    # NB: reflects panels order
    return (ax_s1, ax_s2, ax_s1_hp_t, ax_s1_hp_b,
            ax_event_info, ax_peak_info, ax_s2_hp_t, ax_s2_hp_b,
            ax_ev,
            ax_rec)


def plot_single_event(context: strax.Context,
                      run_id,
                      events,
                      event_number=None,
                      **kwargs):
    """
    Wrapper for event_display

    :param context: strax.context
    :param run_id: run id
    :param events: dataframe / numpy array of events. Should either be
        length 1 or the event_number argument should be provided
    :param event_number: (optional) int, if provided, only show this
        event number
    :param kwargs: kwargs for events_display
    :return: see events_display
    """
    if event_number is not None:
        events = events[events['event_number'] == event_number]
    if len(events) > 1 or len(events) == 0:
        raise ValueError(f'Make sure to provide an event number or a single '
                         f'event. Got {len(events)} events')

    return context.event_display(run_id,
                                 time_range=(events[0]['time'],
                                             events[0]['endtime']),
                                 **kwargs)
