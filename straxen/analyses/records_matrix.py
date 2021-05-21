import warnings

import numba
import numpy as np

import strax
import straxen

DEFAULT_MAX_SAMPLES = 20_000


@straxen.mini_analysis(requires=('records',),
                       warn_beyond_sec=10,
                       default_time_selection='touching')
def records_matrix(records, time_range, seconds_range, config, to_pe,
                   max_samples=DEFAULT_MAX_SAMPLES,
                   ignore_max_sample_warning=False):
    """Return (wv_matrix, times, pms)
      - wv_matrix: (n_samples, n_pmt) array with per-PMT waveform intensity in PE/ns
      - times: time labels in seconds (corr. to rows)
      - pmts: PMT numbers (corr. to columns)
    Both times and pmts have one extra element.

    :param max_samples: Maximum number of time samples. If window and dt
    conspire to exceed this, waveforms will be downsampled.
    :param ignore_max_sample_warning: If True, suppress warning when this happens.

    Example:
        wvm, ts, ys = st.records_matrix(run_id, seconds_range=(1., 1.00001))
        plt.pcolormesh(ts, ys, wvm.T,
                       norm=matplotlib.colors.LogNorm())
        plt.colorbar(label='Intensity [PE / ns]')
    """
    if len(records):
        dt = records[0]['dt']
        samples_per_record = len(records[0]['data'])
    else:
        # Defaults here do not matter, nothing will be plotted anyway
        dt = 10, 110
    record_duration = samples_per_record * dt

    window = time_range[1] - time_range[0]
    if window / dt > max_samples:
        with np.errstate(divide='ignore', invalid='ignore'):
            # Downsample. New dt must be
            #  a) multiple of old dt
            dts = np.arange(0, record_duration + dt, dt).astype(np.int)
            #  b) divisor of record duration
            dts = dts[record_duration / dts % 1 == 0]
            #  c) total samples < max_samples
            dts = dts[window / dts < max_samples]
            if len(dts):
                # Pick lowest dt that satisfies criteria
                dt = dts.min()
            else:
                # Records will be downsampled to single points
                dt = max(record_duration, window // max_samples)
        if not ignore_max_sample_warning:
            warnings.warn(f"Matrix would exceed max_samples {max_samples}, "
                          f"downsampling to dt = {dt} ns.")

    wvm = _records_to_matrix(
        records,
        t0=time_range[0],
        n_channels=config['n_tpc_pmts'],
        dt=dt,
        window=window)
    wvm = wvm.astype(np.float32) * to_pe.reshape(1, -1) / dt

    # Note + 1, so data for sample 0 will range from 0-1 in plot
    ts = (np.arange(wvm.shape[0] + 1) * dt / int(1e9) + seconds_range[0])
    ys = np.arange(wvm.shape[1] + 1)

    return wvm, ts, ys


@straxen.mini_analysis(requires=('raw_records',),
                       warn_beyond_sec=3e-3,
                       default_time_selection='touching')
def raw_records_matrix(context, run_id, raw_records, time_range,
                       ignore_max_sample_warning=False,
                       max_samples=DEFAULT_MAX_SAMPLES,
                       **kwargs):
    # Convert raw to records. We may not be able to baseline correctly
    # at the start of the range due to missing zeroth fragments
    records = strax.raw_to_records(raw_records)
    strax.baseline(records, allow_sloppy_chunking=True)
    strax.zero_out_of_bounds(records)

    return context.records_matrix(run_id=run_id,
                                  records=records,
                                  time_range=time_range,
                                  max_samples=max_samples,
                                  ignore_max_sample_warning=ignore_max_sample_warning,
                                  **kwargs)


@numba.njit
def _records_to_matrix(records, t0, window, n_channels, dt=10):
    n_samples = window // dt
    # Use 32-bit integers, so downsampling saturated samples doesn't
    # cause wraparounds
    # TODO: amplitude bit shift!
    y = np.zeros((n_samples, n_channels),
                 dtype=np.int32)

    if not len(records):
        return y
    samples_per_record = len(records[0]['data'])

    for r in records:
        if r['channel'] > n_channels:
            continue

        if dt >= samples_per_record * r['dt']:
            # Downsample to single sample -> store area
            y[(r['time'] - t0) // dt, r['channel']] += r['area']
            continue

        # Assume out-of-bounds data has been zeroed, so we do not
        # need to do r['data'][:r['length']] here.
        # This simplifies downsampling.
        w = r['data'].astype(np.int32)

        if dt > r['dt']:
            # Downsample
            duration = samples_per_record * r['dt']
            assert duration % dt == 0, "Cannot downsample fractionally"
            # .astype here keeps numba happy ... ??
            w = w.reshape(duration // dt, -1).sum(axis=1).astype(np.int32)

        elif dt < r['dt']:
            raise ValueError("Upsampling not yet implemented")

        (r_start, r_end), (y_start, y_end) = strax.overlap_indices(
            r['time'] // dt, len(w),
            t0 // dt, n_samples)
        # += is paranoid, data in individual channels should not overlap
        # but... https://github.com/AxFoundation/strax/issues/119
        y[y_start:y_end, r['channel']] += w[r_start:r_end]

    return y
