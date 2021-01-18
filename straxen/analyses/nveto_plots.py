import straxen
import strax

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

@straxen.mini_analysis(requires=('raw_records_nv'),
                       warn_beyond_sec=5)
def plot_pulses_nv(context, raw_records, run_id, time_range,
                   plot_hits=False, plot_median=False,
                   max_plots=20, store_pdf=False, path=''):
    """
    Plots nveto pulses for a list of records.

    :param context: Context to be used.
    :param plot_hits: If True plot hit boundaries including the left
        and right extension as orange shaded regions.
    :param plot_median: If true plots pulses sample median as dotted
        line.
    :param max_plots: Limits the number of figures. If you would like
        to plot more pulses you should put the plots in a PDF.
    :param store_pdf: If true figures are put to a PDF instead of
        plotting them to your notebook. The file name is automatically
        generated including the time range and run_id.
    :param path: Relative path where the PDF should be stored. By default
        it is the directory of the notebook.
    """
    # Register records plugin to get settings
    p = context.get_single_plugin(run_id, 'records_nv')
    p.config

    if not raw_records.dtype == np.dtype(strax.raw_record_dtype()):
        raise ValueError('"raw_records" mut be of the raw_records dtype!')

    # Compute strax baseline and baseline_rms:
    # TODO: This is a bit stupid maybe change strax.baseline function?
    records = strax.raw_to_records(raw_records)
    records = strax.sort_by_time(records)
    strax.zero_out_of_bounds(records)
    strax.baseline(records,
                   baseline_samples=p.config['baseline_samples_nv'],
                   flip=True)

    nfigs = 1
    if store_pdf:
        fname = f'pulses_{run_id}_{time_range["time"]}_{time_range["time"]}.pdf'
        fname = os.path.join(path, fname)
        pdf = PdfPages(fname)

    for inds in _yield_pulse_indices(raw_records):
        # Grouped our pulse so now plot:
        rr_pulse = raw_records[inds]
        r_pulse = records[inds]

        fig, axes = straxen.plot_single_pulse(rr_pulse, run_id)

        axes.axhline(rr_pulse[0]['baseline'], ls='dashed',
                     color='k', label=f'D. Bas.: {rr_pulse[0]["baseline"]} ADC')

        baseline = r_pulse[0]['baseline']
        baseline_rms = r_pulse[0]['baseline_rms']
        axes.axhline(baseline, ls='solid',
                     color='k', label=f'Strax Bas. +/-RMS:\n ({baseline:.2f}+/-{baseline_rms:.2f}) ADC')
        xlim = axes.get_xlim()
        axes.fill_between(xlim,
                          [baseline + baseline_rms] * 2,
                          [baseline - baseline_rms] * 2,
                          color='gray', alpha=0.4
                          )
        if plot_median:
            # Plot median if asked.
            # Have to make pulse again:
            pulse = straxen.matplotlib_utils._make_pulse(rr_pulse)
            median = np.median(pulse)
            axes.axhline(median,
                         ls='dotted',
                         color='k',
                         label=f'Median Bas.: {median:.0f} ADC')

            axes.axhline(median - p.config['hit_min_amplitude_nv'],
                         ls='dotted', color='orange'
                         )

        hits = None  # needed for delet if false
        if plot_hits:
            axes.axhline(baseline - p.config['hit_min_amplitude_nv'],
                         color='orange', label='Hitfinder threshold')

            hits = strax.find_hits(r_pulse,
                                   min_amplitude=p.config['hit_min_amplitude_nv']
                                   )
            le, re = p.config['save_outside_hits_nv']
            start = (hits['time'] - r_pulse[0]['time']) / r_pulse[0]['dt'] - le
            end = (strax.endtime(hits) - r_pulse[0]['time']) / r_pulse[0]['dt'] + re

            ylim = axes.get_ylim()
            for s, e in zip(start, end):
                plt.fill_between((s, e), *ylim, alpha=0.2, color='orange')
            axes.set_ylim(*ylim)

        plt.legend()
        axes.set_xlim(*xlim)

        if store_pdf:
            plt.close()
            pdf.savefig(fig)

        nfigs += 1
        if nfigs > max_plots:
            break

    if store_pdf:
        pdf.close()
    del records, hits


def _yield_pulse_indices(records):
    """
    Function which yields indices of records which are within a pulse.

    Note:
        Only finds fragments of the pulse if record_i == 0 is within list
        of records.

    :yields: indices of fragments to make the corresponding pulse.
    """
    # Get record links and find start indicies:
    _, next_ri = strax.record_links(records)
    start_ri = np.where(records['record_i'] == 0)[0]

    # Loop over pulse start_ri, group fragments by pulses yield for plot:
    for ri in start_ri:
        # Buffer for indices:
        inds = []

        tries = 1
        max_tries = 5000
        while ri != -1:
            inds.append(ri)
            ri = next_ri[ri]

            tries += 1
            if tries > max_tries:
                raise ValueError('Tried more than 5000 times to find subsequent record.'
                                 ' Am I stuck in a loop?')
        yield inds
