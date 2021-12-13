from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pytz
import strax
import straxen

export, __all__ = strax.exporter()

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


# Don't be smart with the arguments, since it is a minianalyses we
# need to have all the arguments
@straxen.mini_analysis(requires=('event_info',))
def event_display_simple(context,
                         run_id,
                         events,
                         to_pe,
                         records_matrix=True,
                         s2_fuzz=50,
                         s1_fuzz=0,
                         max_peaks=500,
                         xenon1t=False,
                         display_peak_info=PEAK_DISPLAY_DEFAULT_INFO,
                         display_event_info=EVENT_DISPLAY_DEFAULT_INFO,
                         s1_hp_kwargs=None,
                         s2_hp_kwargs=None,
                         event_time_limit=None,
                         plot_all_positions=True,
                         ):
    """
    {event_docs}
    {event_returns}
    """
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    grid = plt.GridSpec(2, 3, hspace=0.5)
    axes = dict()
    axes["ax_s1"] = fig.add_subplot(grid[0, 0])
    axes["ax_s2"] = fig.add_subplot(grid[0, 1])
    axes["ax_s2_hp_t"] = fig.add_subplot(grid[0, 2])
    axes["ax_ev"] = fig.add_subplot(grid[1, :])

    return _event_display(context,
                          run_id,
                          events,
                          to_pe,
                          axes=axes,
                          records_matrix=records_matrix,
                          s2_fuzz=s2_fuzz,
                          s1_fuzz=s1_fuzz,
                          max_peaks=max_peaks,
                          xenon1t=xenon1t,
                          display_peak_info=display_peak_info,
                          display_event_info=display_event_info,
                          s1_hp_kwargs=s1_hp_kwargs,
                          s2_hp_kwargs=s2_hp_kwargs,
                          event_time_limit=event_time_limit,
                          plot_all_positions=plot_all_positions,
                          )


# Don't be smart with the arguments, since it is a minianalyses we
# need to have all the arguments
@straxen.mini_analysis(requires=('event_info',))
def event_display(context,
                  run_id,
                  events,
                  to_pe,
                  records_matrix=True,
                  s2_fuzz=50,
                  s1_fuzz=0,
                  max_peaks=500,
                  xenon1t=False,
                  display_peak_info=PEAK_DISPLAY_DEFAULT_INFO,
                  display_event_info=EVENT_DISPLAY_DEFAULT_INFO,
                  s1_hp_kwargs=None,
                  s2_hp_kwargs=None,
                  event_time_limit=None,
                  plot_all_positions=True,
                  ):
    """
    {event_docs}
    {event_returns}
    """
    if records_matrix not in ('raw', True, False):
        raise ValueError('Choose either "raw", True or False for records_matrix')
    if ((records_matrix == 'raw' and not context.is_stored(run_id, 'raw_records')) or
            (isinstance(records_matrix, bool) and not context.is_stored(run_id,
                                                                        'records'))):  # noqa
        print("(raw)records not stored! Not showing records_matrix")
        records_matrix = False
    # Convert string to int to allow plots to be enlarged for extra panel
    _rr_resize_int = int(bool(records_matrix))

    fig = plt.figure(figsize=(25, 21 if _rr_resize_int else 16),
                     facecolor='white')
    grid = plt.GridSpec((2 + _rr_resize_int), 1, hspace=0.1 + 0.1 * _rr_resize_int,
                        height_ratios=[1.5, 0.5, 0.5][:2 + _rr_resize_int]
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
    ax_rec = None

    # (raw)records matrix (optional)
    if records_matrix:
        gss_2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid[2])
        ax_rec = fig.add_subplot(gss_2[0])
    axes = dict(
        ax_s1=ax_s1,
        ax_s2=ax_s2,
        ax_s1_hp_t=ax_s1_hp_t,
        ax_s1_hp_b=ax_s1_hp_b,
        ax_event_info=ax_event_info,
        ax_peak_info=ax_peak_info,
        ax_s2_hp_t=ax_s2_hp_t,
        ax_s2_hp_b=ax_s2_hp_b,
        ax_ev=ax_ev,
        ax_rec=ax_rec)

    return _event_display(context,
                          run_id,
                          events,
                          to_pe,
                          axes=axes,
                          records_matrix=records_matrix,
                          s2_fuzz=s2_fuzz,
                          s1_fuzz=s1_fuzz,
                          max_peaks=max_peaks,
                          xenon1t=xenon1t,
                          display_peak_info=display_peak_info,
                          display_event_info=display_event_info,
                          s1_hp_kwargs=s1_hp_kwargs,
                          s2_hp_kwargs=s2_hp_kwargs,
                          event_time_limit=event_time_limit,
                          plot_all_positions=plot_all_positions,
                          )


def _event_display(context,
                   run_id,
                   events,
                   to_pe,
                   axes=None,
                   records_matrix=True,
                   s2_fuzz=50,
                   s1_fuzz=0,
                   max_peaks=500,
                   xenon1t=False,
                   display_peak_info=PEAK_DISPLAY_DEFAULT_INFO,
                   display_event_info=EVENT_DISPLAY_DEFAULT_INFO,
                   s1_hp_kwargs=None,
                   s2_hp_kwargs=None,
                   event_time_limit=None,
                   plot_all_positions=True,
                   ):
    """{event_docs}
    :param axes: if a dict of matplotlib axes (w/ same keys as below,
        and empty/None for panels not filled)
    {event_returns} 
    """
    if len(events) != 1:
        raise ValueError(f'Found {len(events)} only request one')
    event = events[0]

    if axes is None:
        raise ValueError(f'No axes provided')
    ax_s1 = axes.get("ax_s1", None)
    ax_s2 = axes.get("ax_s2", None)
    ax_s1_hp_t = axes.get("ax_s1_hp_t", None)
    ax_s1_hp_b = axes.get("ax_s1_hp_b", None)
    ax_s2_hp_t = axes.get("ax_s2_hp_t", None)
    ax_s2_hp_b = axes.get("ax_s2_hp_b", None)
    ax_event_info = axes.get("ax_event_info", None)
    ax_peak_info = axes.get("ax_peak_info", None)
    ax_ev = axes.get("ax_ev", None)
    ax_rec = axes.get("ax_rec", None)

    # titles
    for ax, title in zip([ax_s1, ax_s1_hp_t, ax_s1_hp_b,
                          ax_s2, ax_s2_hp_t, ax_s2_hp_b,
                          ax_event_info, ax_peak_info],
                         ["Main S1", "S1 top", "S1 bottom",
                          "Main S2", "S2 top", "S2 bottom",
                          "Event info", "Peak info"]):
        if ax is not None:
            ax.set_title(title)

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
                           dead_pmts=np.argwhere(to_pe == 0),
                           cmap=color_map)
        # update s1 & S2 hit pattern kwargs with _common_opt if not
        # specified by the user
        for k, v in _common_opt.items():
            if k not in hp_opt:
                hp_opt[k] = v

    # S1
    if events['s1_area'] != 0:
        if ax_s1 is not None:
            plt.sca(ax_s1)
            context.plot_peaks(run_id,
                               time_range=(events['s1_time'] - s1_fuzz,
                                           events['s1_endtime'] + s1_fuzz),
                               single_figure=False)

        # Hit pattern plots
        area = context.get_array(run_id, 'peaklets',
                                 time_range=(events['s1_time'],
                                             events['s1_endtime']),
                                 keep_columns=('area_per_channel', 'time', 'dt', 'length'),
                                 progress_bar=False,
                                 )
        for ax, array in ((ax_s1_hp_t, 'top'), (ax_s1_hp_b, 'bottom')):
            if ax is not None:
                plt.sca(ax)
                straxen.plot_on_single_pmt_array(c=np.sum(area['area_per_channel'], axis=0),
                                                 array_name=array,
                                                 **s1_hp_kwargs)
                # Mark reconstructed position
                plt.scatter(event['x'], event['y'], marker='X', s=100, c='k')

    # S2
    if event['s2_area'] != 0:
        if ax_s2 is not None:
            plt.sca(ax_s2)
            context.plot_peaks(run_id,
                               time_range=(events['s2_time'] - s2_fuzz,
                                           events['s2_endtime'] + s2_fuzz),
                               single_figure=False)

        # Hit pattern plots
        area = context.get_array(run_id, 'peaklets',
                                 time_range=(events['s2_time'],
                                             events['s2_endtime']),
                                 keep_columns=('area_per_channel', 'time', 'dt', 'length'),
                                 progress_bar=False,
                                 )
        for axi, (ax, array) in enumerate([(ax_s2_hp_t, 'top'), (ax_s2_hp_b, 'bottom')]):
            if ax is not None:
                plt.sca(ax)
                straxen.plot_on_single_pmt_array(c=np.sum(area['area_per_channel'], axis=0),
                                                 array_name=array,
                                                 **s2_hp_kwargs)
                # Mark reconstructed position (corrected)
                plt.scatter(event['x'], event['y'], marker='X', s=100, c='k')
                if not xenon1t and axi == 0 and plot_all_positions:
                    _scatter_rec(event)

    # Fill panels with peak/event info
    for it, (ax, labels_and_unit) in enumerate([(ax_event_info, display_event_info),
                                                (ax_peak_info, display_peak_info)]):
        if ax is not None:
            for i, (_lab, _unit) in enumerate(labels_and_unit):
                coord = 0.01, 0.9 - 0.9 * i / len(labels_and_unit)
                ax.text(*coord, _lab[:24], va='top', zorder=-10)
                ax.text(coord[0] + 0.5, coord[1],
                        _unit.format(v=event[_lab]), va='top', zorder=-10)
                # Remove axes and labels from panel
                ax.set_xticks([])
                ax.set_yticks([])
                _ = [s.set_visible(False) for s in ax.spines.values()]

    # Plot peaks in event
    ev_range = None
    if ax_ev is not None:
        plt.sca(ax_ev)
        if event_time_limit is None:
            time_range = (events['time'], events['endtime'])
        else:
            time_range = event_time_limit

        context.plot_peaks(run_id,
                           time_range=time_range,
                           show_largest=max_peaks,
                           single_figure=False)
        ev_range = plt.xlim()

    if records_matrix and ax_rec is not None:
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
        if ev_range is not None:
            plt.xlim(*ev_range)

    # Final tweaks
    if ax_s2 is not None:
        ax_s1.tick_params(axis='x', rotation=45)
    if ax_s2 is not None:
        ax_s1.tick_params(axis='x', rotation=45)
    if ax_ev is not None:
        ax_ev.tick_params(axis='x', rotation=0)
    title = (f'Run {run_id}. Time '
             f'{str(events["time"])[:-9]}.{str(events["time"])[-9:]}\n'
             f'{datetime.fromtimestamp(event["time"] / 1e9, tz=pytz.utc)}')
    plt.suptitle(title, y=0.95)
    # NB: reflects panels order
    return (ax_s1, ax_s2, ax_s1_hp_t, ax_s1_hp_b,
            ax_event_info, ax_peak_info, ax_s2_hp_t, ax_s2_hp_b,
            ax_ev,
            ax_rec)


@export
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


def _scatter_rec(_event,
                 recs=None,
                 scatter_kwargs=None,
                 ):
    """Convenient wrapper to show posrec of three algorithms for xenonnt"""
    if recs is None:
        recs = ('mlp', 'cnn', 'gcn')
    elif len(recs) > 5:
        raise ValueError("I only got five markers/colors")
    if scatter_kwargs is None:
        scatter_kwargs = {}
    scatter_kwargs.setdefault('s', 100)
    scatter_kwargs.setdefault('alpha', 0.8)
    shapes = ('v', '^', '>', '<', '*', 'D', "P")
    colors = ('brown', 'orange', 'lightcoral', 'gold', 'lime', 'crimson')
    for _i, _r in enumerate(recs):
        x, y = _event[f's2_x_{_r}'], _event[f's2_y_{_r}']
        if np.isnan(x) or np.isnan(y):
            continue
        plt.scatter(x, y,
                    marker=shapes[_i],
                    c=colors[_i],
                    label=_r.upper(),
                    **scatter_kwargs,
                    )
    plt.legend(loc='best', fontsize="x-small", markerscale=0.5)


# Event display docstrings.
# Let's add them to the corresponding functions

event_docs = """
Make a waveform-display of a given event. Requires events, peaks and
    peaklets (optionally: records). NB: time selection should return
    only one event!

:param context: strax.Context provided by the minianalysis wrapper
:param run_id: run-id of the event
:param events: events, provided by the minianalysis wrapper
:param to_pe: gains, provided by the minianalysis wrapper
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
:param event_time_limit = overrides x-axis limits of event
    plot
:param plot_all_positions if True, plot best-fit positions
    from all posrec algorithms
"""
event_returns = """
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

# Add the same docstring to each of these functions
for event_function in (event_display, event_display_simple, _event_display):
    doc = event_function.__doc__
    if doc is not None:
        event_function.__doc__ = doc.format(event_docs=event_docs,
                                            event_returns=event_returns)
