import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import strax
import straxen
from datetime import datetime
import pytz

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


def _scatter_rec(_event,
                 recs=None,
                 scatter_kwargs=None):
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
                    **scatter_kwargs
                    )
    plt.legend(loc='best', fontsize="x-small", markerscale=0.5)


@straxen.mini_analysis(requires=('event_info', 'event_posrec_many'))
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
                  ):
    """
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
        (isinstance(records_matrix, bool) and not context.is_stored(run_id, 'records'))):   # noqa
        print("(raw)records not stored! Not showing records_matrix")
        records_matrix = False
    if not context.is_stored(run_id, 'peaklets'):
        raise strax.DataNotAvailable(f'peaklets not available for {run_id}')

    # Convert string to int to allow plots to be enlarged for extra panel
    _rr_resize_int = int(bool(records_matrix))
    fig = plt.figure(figsize=(25, 21 if _rr_resize_int else 16),
                     facecolor='white')
    grid = plt.GridSpec((2 + _rr_resize_int), 1, hspace=0.1+0.1*_rr_resize_int,
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
                           dead_pmts=np.argwhere(to_pe == 0),
                           cmap=color_map)
        # update s1 & S2 hit pattern kwargs with _common_opt if not
        # specified by the user
        for k, v in _common_opt.items():
            if k not in hp_opt:
                hp_opt[k] = v

    # S1
    if events['s1_area'] != 0:
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
            plt.sca(ax)
            straxen.plot_on_single_pmt_array(c=np.sum(area['area_per_channel'], axis=0),
                                             array_name=array,
                                             **s1_hp_kwargs)
            # Mark reconstructed position
            plt.scatter(event['x'], event['y'], marker='X', s=100, c='r')

    # S2
    if event['s2_area'] != 0:
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
            plt.sca(ax)
            straxen.plot_on_single_pmt_array(c=np.sum(area['area_per_channel'], axis=0),
                                             array_name=array,
                                             **s2_hp_kwargs)
            # Mark reconstructed position (corrected)
            plt.scatter(event['x'], event['y'], marker='X', s=100, c='r')
            if not xenon1t and axi == 0:
                _scatter_rec(event)

    # Fill panels with peak/event info
    for it, (ax, labels_and_unit) in enumerate([(ax_event_info, display_event_info),
                                                (ax_peak_info, display_peak_info)]):
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
