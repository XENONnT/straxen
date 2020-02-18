import numba
import numpy as np

import strax
import straxen


@straxen.mini_analysis(requires=('records',),
                       warn_beyond_sec=5e-3,
                       default_time_selection='touching')
def records_matrix(records, time_range, seconds_range, to_pe):
    """Return (wv_matrix, times, pms)
      - wv_matrix: (n_samples, n_pmt) array with per-PMT waveform intensity in PE/ns
      - times: time labels in seconds (corr. to rows)
      - pmts: PMT numbers (corr. to columns)
    Both times and pmts have one extra element.

    Example:
        wvm, ts, ys = st.records_matrix(run_id, seconds_range=(1., 1.00001))
        plt.pcolormesh(ts, ys, wvm.T,
                       norm=matplotlib.colors.LogNorm())
        plt.colorbar(label='Intensity [PE / ns]')
    """
    if len(records):
        dt = records[0]['dt']
    else:
        dt = 10  # But it doesn't matter, nothing will be plotted anyway

    for x in time_range:
        # TODO: this check should probably move to strax!
        if not isinstance(x, (int, np.integer)):
            raise ValueError(f"Time range must consist of integers, "
                             f"found a {type(x)}")

    wvm = _records_to_matrix(
        records,
        t0=time_range[0],
        window=time_range[1] - time_range[0])
    wvm = wvm.astype(np.float32) * to_pe.reshape(1, -1) / dt

    # Note + 1, so data for sample 0 will range from 0-1 in plot
    ts = (np.arange(wvm.shape[0] + 1) * dt / int(1e9) + seconds_range[0])
    ys = np.arange(wvm.shape[1] + 1)

    return wvm, ts, ys


@straxen.mini_analysis(requires=('raw_records',),
                       warn_beyond_sec=3e-3,
                       default_time_selection='touching')
def raw_records_matrix(context, run_id, raw_records, time_range):
    return context.records_matrix(run_id=run_id, records=raw_records, time_range=time_range)


@numba.njit
def _records_to_matrix(records, t0, window, n_channels=straxen.n_tpc_pmts, dt=10):
    n_samples = window // dt
    y = np.zeros((n_samples, n_channels),
                 dtype=np.int32)

    for r in records:
        if r['channel'] > n_channels:
            continue
        (r_start, r_end), (y_start, y_end) = strax.overlap_indices(
            r['time'] // dt, r['length'],
            t0 // dt, n_samples)
        # += is paranoid, data in individual channels should not overlap
        # but... https://github.com/AxFoundation/strax/issues/119
        y[y_start:y_end, r['channel']] += r['data'][r_start:r_end]
    return y
