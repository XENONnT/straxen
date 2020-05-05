import numpy as np
from multihist import Hist1d, Histdd
import matplotlib.pyplot as plt

import straxen


@straxen.mini_analysis(requires=('peak_basics',))
def plot_peaks_aft_histogram(
        context, run_id, peaks,
        pe_bins=np.logspace(0, 7, 120),
        rt_bins=np.geomspace(2, 1e5, 120),
        extra_labels=tuple(),
        rate_range=(1e-4, 1),
        aft_range=(0, .85),
        figsize=(14, 5)):
    """Plot side-by-side (area, width) histograms of the peak rate
    and mean area fraction top.

    :param pe_bins: Array of bin edges for the peak area dimension [PE]
    :param rt_bins: array of bin edges for the rise time dimension [ns]
    :param extra_labels: List of (area, risetime, text, color) extra labels
    to put on the plot
    :param rate_range: Range of rates to show [peaks/(bin*s)]
    :param aft_range: Range of mean S1 area fraction top / bin to show
    :param figsize: Figure size to use
    """
    livetime_sec = straxen.get_livetime_sec(context, run_id, peaks)

    mh = Histdd(peaks,
                dimensions=(
                    ('area', pe_bins),
                    ('range_50p_area', rt_bins),
                    ('area_fraction_top', np.linspace(0, 1, 100))
                ))

    f, axes = plt.subplots(1, 2, figsize=figsize)

    def std_axes():
        plt.gca().set_facecolor('k')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("Area [PE]")
        plt.ylabel("Range 50% area [ns]")
        labels = [
            (12, 8, "AP?", 'white'),
            (3, 150, "1PE\npileup", 'gray'),

            (30, 200, "1e", 'gray'),
            (100, 1000, "n-e", 'w'),
            (2000, 2e4, "Train", 'gray'),

            (1200, 50, "S1", 'w'),
            (45e3, 60, "αS1", 'w'),

            (2e5, 800, "S2", 'w'),
        ] + list(extra_labels)

        for x, w, text, color in labels:
            plt.text(x, w, text, color=color,
                     verticalalignment='center',
                     horizontalalignment='center')

    plt.sca(axes[0])
    (mh / livetime_sec).sum(axis=2).plot(
        log_scale=True,
        vmin=rate_range[0], vmax=rate_range[1],
        colorbar_kwargs=dict(extend='both'),
        cblabel='Peaks / (bin * s)')
    std_axes()

    plt.sca(axes[1])
    mh.average(axis=2).plot(
        vmin=aft_range[0], vmax=aft_range[1],
        colorbar_kwargs=dict(extend='max'),
        cmap=plt.cm.jet, cblabel='Mean area fraction top')

    std_axes()
    plt.tight_layout()


@straxen.mini_analysis(requires=['event_info'])
def event_scatter(context, run_id, events,
                  show_single=True,
                  s=10,
                  color_range=(None, None),
                  color_dim='s1_area_fraction_top',
                  figsize=(7, 5)):
    """Plot a (cS1, cS2) event scatter plot

    :param show_single: Show events with only S1s or only S2s just besides
    the axes.
    :param s: Scatter size
    :param color_dim: Dimension to use for the color. Must be in event_info.
    :param color_range: Minimum and maximum color value to show.
    :param figsize: (w, h) figure size to use, or leave None to not make a
    new matplotlib figure.
    """
    if figsize is not None:
        plt.figure(figsize=figsize)
    if color_dim == 's1_area_fraction_top' and color_range == (None, None):
        color_range = (0, 0.3)

    plt.scatter(np.nan_to_num(events['cs1']).clip(.9, None),
                np.nan_to_num(events['cs2']).clip(.9, None),
                clip_on=not show_single,
                c=events[color_dim],
                vmin=color_range[0], vmax=color_range[1],
                s=s,
                cmap=plt.cm.jet,
                marker='.', edgecolors='none')

    plt.xlabel('cS1 [PE]')
    plt.xscale('log')
    plt.xlim(1, None)

    plt.ylabel('cS2 [PE]')
    plt.yscale('log')
    plt.ylim(1, None)

    p = context.get_single_plugin(run_id, 'energy_estimates')
    ax = plt.gca()
    el_lim = p.cs1_to_e(np.asarray(ax.get_xlim()))
    ec_lim = p.cs2_to_e(np.asarray(ax.get_ylim()))

    ax2 = ax.twiny()
    ax2.set_xlim(*el_lim)
    ax2.set_xscale('log')
    ax2.set_xlabel("E_light [keVee]")

    ax3 = ax2.twinx()
    ax3.set_ylim(*ec_lim)
    ax3.set_yscale('log')
    ax3.set_ylabel("E_charge [keVee]")

    plt.sca(ax3)
    plt.plot(el_lim, el_lim, c='k', alpha=0.2)
    x = np.geomspace(*el_lim, num=1000)
    e_label = 1.2e-3
    for e_const, label in [
            (0.1, ''), (1, '1\nkeV'), (10, '10\nkeV'),
            (100, '100\nkeV'), (1e3, '1\nMeV'), (1e4, '')]:
        plt.plot(x, e_const - x, c='k', alpha=0.2)
        plt.text(e_const - e_label, e_label, label,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                 horizontalalignment='center', verticalalignment='center',
                 color='k', alpha=0.5)

    plt.sca(ax)
    if color_range[0] is None:
        extend = 'neither' if color_range[1] is None else 'max'
    else:
        extend = 'min' if color_range[1] is None else 'both'
    if color_dim == 's1_area_fraction_top':
        plt.colorbar(label="S1 area fraction top",
                     extend=extend,
                     ax=[ax, ax3])
    else:
        plt.colorbar(label=color_dim,
                     extend=extend,
                     ax=[ax, ax3])

@straxen.mini_analysis(requires=('event_info',))
def plot_energy_spectrum(
        events,
        color='b', label=None,
        unit=None, exposure_kg_sec=None,
        error_alpha=0.5, errors='fc',
        n_bins=100, min_energy=1, max_energy=100, geomspace=True):
    """Plot an energy spectrum histogram, with 1 sigma
    Poisson confidence intervals around it.

    :param exposure_kg_sec: Exposure in kg * sec
    :param unit: Unit to plot spectrum in. Can be either:
      - events (events per bin)
      - kg_day_kev (events per kg day keV)
      - tonne_day_kev (events per tonne day keV)
      - tonne_year_kev (events per tonne year keV)
    Defaults to kg_day_kev if exposure_kg_sec is provided,
    otherwise events.

    :param min_energy: Minimum energy of the histogram
    :param max_energy: Maximum energy of the histogram
    :param geomspace: If True, will use a logarithmic energy binning.
    Otherwise will use a linear scale.
    :param n_bins: Number of energy bins to use
    
    :param color: Color to plot in
    :param label: Label for the line
    :param error_alpha: Alpha value for the statistical error band
    :param errors: Type of errors to draw, passed to 'errors'
    argument of Hist1d.plot.
    """
    if unit is None:
        if exposure_kg_sec is not None:
            unit = 'kg_day_kev'
        else:
            unit = 'events'       
    
    h = Hist1d(events['e_ces'],
               bins=(np.geomspace if geomspace else np.linspace)(
                   min_energy, max_energy, n_bins))

    if unit == 'events':
        scale, ylabel = 1, 'Events per bin'
    else:
        exposure_kg_day = exposure_kg_sec / (3600 * 24)
        if unit == 'kg_day_kev':
            scale = exposure_kg_day
            ylabel = 'Events / (kg day keV)'
        elif unit == 'tonne_day_kev':
            scale = exposure_kg_day / 1000
            ylabel = 'Events / (tonne day keV)'
        elif unit == 'tonne_year_kev':
            scale = exposure_kg_day / 1000
            ylabel = 'Events / (tonne year keV)'
        else:
            raise ValueError(f"Invalid unit {unit}")
        scale *= h.bin_volumes()
    
    h.plot(errors=errors,
           error_style='band',
           color=color,
           label=label,
           linewidth=1,
           scale_histogram_by=1/scale,
           error_alpha=error_alpha)
    plt.yscale('log')
    if geomspace:
        straxen.log_x(min_energy, max_energy, scalar_ticks=True)
    else:
        plt.xlim(min_energy, max_energy)
    plt.ylabel(ylabel)
    plt.xlabel("Energy [keV_ee], CES")


@straxen.mini_analysis(requires=('peak_basics',))
def plot_peak_classification(peaks, s=1):
    """Make an (area, rise_time) scatter plot of peaks
    :param s: Size of dot for each peak
    """
    for cl, color in enumerate('kbg'):
        d = peaks[peaks['type'] == cl]
        plt.scatter(d['area'], d['rise_time'], c=color,
                    s=s, marker='.', edgecolors='none',
                    label={0: 'Unknown', 1: 'S1', 2: 'S2'}[cl])
    plt.legend(loc='lower right', markerscale=10)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, 2e6)
    plt.ylim(3, 1e4)
    plt.xlabel("Area [PE]")
    plt.ylabel("Rise time [ns]")
