import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def log_y(a=None, b=None, scalar_ticks=True, tick_at=None):
    """Make the y axis use a log scale from a to b"""
    plt.yscale('log')
    if a is not None:
        if b is None:
            a, b = a[0], a[-1]
        ax = plt.gca()
        plt.ylim(a, b)
        if scalar_ticks:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.set_yticks(logticks(a, b, tick_at))


def log_x(a=None, b=None, scalar_ticks=True, tick_at=None):
    """Make the x axis use a log scale from a to b"""
    plt.xscale('log')
    if a is not None:
        if b is None:
            a, b = a[0], a[-1]
        plt.xlim(a, b)
        ax = plt.gca()
        if scalar_ticks:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.set_xticks(logticks(a, b, tick_at))


def logticks(tmin, tmax=None, tick_at=None):
    if tick_at is None:
        tick_at = (1, 2, 5, 10)
    a, b = np.log10([tmin, tmax])
    a = np.floor(a)
    b = np.ceil(b)
    ticks = np.sort(np.unique(np.outer(
        np.array(tick_at), 
        10.**np.arange(a, b)).ravel()))
    ticks = ticks[(tmin <= ticks) & (ticks <= tmax)]
    return ticks


def quiet_tight_layout():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()