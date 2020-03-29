import numba
import numpy as np

import strax
import straxen
export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        'to_pe_file',
        default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',    # noqa
        help='URL of the to_pe conversion factors'),
    strax.Option(
        'baseline_samples',
        default=40,
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline'),

    # Tail veto options
    strax.Option(
        'tail_veto_threshold',
        default=int(1e5),
        help=("Minimum peakarea in PE to trigger tail veto."
              "Set to None, 0 or False to disable veto.")),
    strax.Option(
        'tail_veto_duration',
        default=int(3e6),
        help="Time in ns to veto after large peaks"),
    strax.Option(
        'tail_veto_resolution',
        default=int(1e3),
        help="Time resolution in ns for pass-veto waveform summation"),
    strax.Option(
        'tail_veto_pass_fraction',
        default=0.05,
        help="Pass veto if maximum amplitude above max * fraction"),
    strax.Option(
        'tail_veto_pass_extend',
        default=3,
        help="Extend pass veto by this many samples (tail_veto_resolution!)"),

    # PMT pulse processing options
    strax.Option(
        'pmt_pulse_filter',
        default=(0.012, -0.119,
                 2.435, -1.271, 0.357, -0.174, -0., -0.036,
                 -0.028, -0.019, -0.025, -0.013, -0.03, -0.039,
                 -0.005, -0.019, -0.012, -0.015, -0.029, 0.024,
                 -0.007, 0.007, -0.001, 0.005, -0.002, 0.004, -0.002),
        help='Linear filter to apply to pulses, will be normalized.'),
    strax.Option(
        'save_outside_hits',
        default=(3, 3),
        help='Save (left, right) samples besides hits; cut the rest'),

    strax.Option(
        'hit_min_amplitude',
        default=straxen.adc_thresholds(),
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_tpc_pmts, or a number.'),

    strax.Option(
        'n_tpc_pmts',
        default=straxen.n_tpc_pmts,
        help='Number of TPC PMTs'),
    strax.Option(
        'n_diagnostic_pmts',
        default=254 - 248,
        help='Number of diagnostic PMTs'),

    strax.Option(
        'check_raw_record_overlaps',
        default=True, track=False,
        help='Crash if any of the pulses in raw_records overlap with others '
             'in the same channel'),

)
class PulseProcessing(strax.Plugin):
    """
    1. Split raw_records into:
     - tpc_records
     - diagnostic_records
     - aqmon_records

    For TPC records, apply basic processing:
    1. Flip, baseline, and integrate the waveform
    2. Apply software HE veto after high-energy peaks.
    3. Find hits, apply linear filter, and zero outside hits.
    """
    __version__ = '0.1.1'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'raw_records'

    provides = ('records', 'diagnostic_records', 'aqmon_records',
                'veto_regions', 'pulse_counts')
    data_kind = {k: k for k in provides}

    def infer_dtype(self):
        # Get record_length from the plugin making raw_records
        self.record_length = strax.record_length_from_dtype(
            self.deps['raw_records'].dtype_for('raw_records'))

        dtype = dict()
        for p in self.provides:
            if 'records' in p:
                dtype[p] = strax.record_dtype(self.record_length)
        dtype['veto_regions'] = strax.hit_dtype
        dtype['pulse_counts'] = pulse_count_dtype(self.config['n_tpc_pmts'])

        return dtype

    def setup(self):
        self.to_pe = straxen.get_to_pe(self.run_id, self.config['to_pe_file'])

    def compute(self, raw_records):
        if self.config['check_raw_record_overlaps']:
            check_overlaps(raw_records, n_channels=3000)

        # Convert everything to the records data type -- adds extra fields.
        records = strax.raw_to_records(raw_records)
        del raw_records

        # Split off non-TPC records
        r, other = channel_split(records, self.config['n_tpc_pmts'])
        diagnostic_records, aqmon_records = channel_split(
            other,
            self.config['n_tpc_pmts'] + self.config['n_diagnostic_pmts'])
        del records

        # From here on we only work on TPC records, stored in r

        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        # TODO: better to throw an error if something is nonzero
        strax.zero_out_of_bounds(r)

        strax.baseline(r,
                       baseline_samples=self.config['baseline_samples'],
                       flip=True)

        strax.integrate(r)

        pulse_counts = count_pulses(r, self.config['n_tpc_pmts'])

        if self.config['tail_veto_threshold'] and len(r):
            r, r_vetoed, veto_regions = software_he_veto(
                r, self.to_pe,
                area_threshold=self.config['tail_veto_threshold'],
                veto_length=self.config['tail_veto_duration'],
                veto_res=self.config['tail_veto_resolution'],
                pass_veto_fraction=self.config['tail_veto_pass_fraction'],
                pass_veto_extend=self.config['tail_veto_pass_extend'])

            # In the future, we'll probably want to sum the waveforms
            # inside the vetoed regions, so we can still save the "peaks".
            del r_vetoed

        else:
            veto_regions = np.zeros(0, dtype=strax.hit_dtype)

        if len(r):
            # Find hits
            # -- before filtering,since this messes with the with the S/N
            hits = strax.find_hits(
                r,
                min_amplitude=self.config['hit_min_amplitude'])

            if self.config['pmt_pulse_filter']:
                # Filter to concentrate the PMT pulses
                strax.filter_records(
                    r, np.array(self.config['pmt_pulse_filter']))

            le, re = self.config['save_outside_hits']
            r = strax.cut_outside_hits(r, hits,
                                       left_extension=le,
                                       right_extension=re)

            # Probably overkill, but just to be sure...
            strax.zero_out_of_bounds(r)

        return dict(records=r,
                    diagnostic_records=diagnostic_records,
                    aqmon_records=aqmon_records,
                    pulse_counts=pulse_counts,
                    veto_regions=veto_regions)


##
# Software HE Veto
##

@export
def software_he_veto(records, to_pe,
                     area_threshold=int(1e5),
                     veto_length=int(3e6),
                     veto_res=int(1e3), pass_veto_fraction=0.01,
                     pass_veto_extend=3):
    """Veto veto_length (time in ns) after peaks larger than
    area_threshold (in PE).

    Further large peaks inside the veto regions are still passed:
    We sum the waveform inside the veto region (with time resolution
    veto_res in ns) and pass regions within pass_veto_extend samples
    of samples with amplitude above pass_veto_fraction times the maximum.

    :returns: (preserved records, vetoed records, veto intervals).

    :param records: PMT records
    :param to_pe: ADC to PE conversion factors for the channels in records.
    :param area_threshold: Minimum peak area to trigger the veto.
    Note we use a much rougher clustering than in later processing.
    :param veto_length: Time in ns to veto after the peak
    :param veto_res: Resolution of the sum waveform inside the veto region.
    Do not make too large without increasing integer type in some strax
    dtypes...
    :param pass_veto_fraction: fraction of maximum sum waveform amplitude to
    trigger veto passing of further peaks
    :param pass_veto_extend: samples to extend (left and right) the pass veto
    regions.
    """
    veto_res = int(veto_res)
    if veto_res > np.iinfo(np.int16).max:
        raise ValueError("Veto resolution does not fit 16-bit int")
    veto_length = np.ceil(veto_length / veto_res).astype(np.int) * veto_res
    veto_n = int(veto_length / veto_res) + 1

    # 1. Find large peaks in the data.
    # This will actually return big agglomerations of peaks and their tails
    peaks = strax.find_peaks(
        records, to_pe,
        gap_threshold=1,
        left_extension=0,
        right_extension=0,
        min_channels=100,
        min_area=area_threshold,
        result_dtype=strax.peak_dtype(n_channels=len(to_pe),
                                      n_sum_wv_samples=veto_n))

    # 2a. Set 'candidate regions' at these peaks. These should:
    #  - Have a fixed maximum length (else we can't use the strax hitfinder on them)
    #  - Never extend beyond the current chunk
    #  - Do not overlap
    # TODO: pass the chunk endtime! For now we just use the last record endtime
    veto_start = peaks['time']
    veto_end = np.clip(peaks['time'] + veto_length,
                       None,
                       strax.endtime(records[-1]))
    veto_end[:-1] = np.clip(veto_end[:-1], None, veto_start[1:])

    # 2b. Convert these into strax record-like objects
    # Note the waveform is float32 though (it's a summed waveform)
    regions = np.zeros(
        len(veto_start),
        dtype=strax.interval_dtype + [
            ("data", (np.float32, veto_n)),
            ("baseline", np.float32),
            ("baseline_rms", np.float32),
            ("reduction_level", np.int64),
            ("record_i", np.int64),
            ("pulse_length", np.int64),
        ])
    regions['time'] = veto_start
    regions['length'] = (veto_end - veto_start) // veto_n
    regions['pulse_length'] = veto_n
    regions['dt'] = veto_res

    if not len(regions):
        # No veto anywhere in this data
        return records, records[:0], np.zeros(0, strax.hit_dtype)

    # 3. Find pass_veto regios with big peaks inside the veto regions.
    # For this we compute a rough sum waveform (at low resolution,
    # without looping over the pulse data)
    rough_sum(regions, records, to_pe, veto_n, veto_res)
    regions['data'] /= np.max(regions['data'], axis=1)[:, np.newaxis]

    pass_veto = strax.find_hits(regions, min_amplitude=pass_veto_fraction)

    # 4. Extend these by a few samples and inverse to find veto regions
    regions['data'] = 1
    regions = strax.cut_outside_hits(
        regions,
        pass_veto,
        left_extension=pass_veto_extend,
        right_extension=pass_veto_extend)
    regions['data'] = 1 - regions['data']
    veto = strax.find_hits(regions, min_amplitude=1)
    # Do not remove very tiny regions
    veto = veto[veto['length'] > 2 * pass_veto_extend]

    # 5. Apply the veto and return results
    veto_mask = strax.fully_contained_in(records, veto) == -1
    return tuple(list(_mask_and_not(records, veto_mask)) + [veto])


@numba.njit(cache=True, nogil=True)
def rough_sum(regions, records, to_pe, n, dt):
    """Compute ultra-rough sum waveforms for regions, assuming:
     - every record is a single peak at its first sample
     - all regions have the same length and dt
    and probably not carying too much about boundaries
    """
    if not len(regions) or not len(records):
        return

    # dt and n are passed explicitly to avoid overflows/wraparounds
    # related to the small dt integer type

    peak_i = 0
    r_i = 0
    while (peak_i <= len(regions) - 1) and (r_i <= len(records) - 1):

        p = regions[peak_i]
        l = p['time']
        r = l + n * dt

        while True:
            if r_i > len(records) - 1:
                # Scan ahead until records contribute
                break
            t = records[r_i]['time']
            if t >= r:
                break
            if t >= l:
                index = int((t - l) // dt)
                regions[peak_i]['data'][index] += (
                        records[r_i]['area'] * to_pe[records[r_i]['channel']])
            r_i += 1
        peak_i += 1


##
# Pulse counting
##

@export
def pulse_count_dtype(n_channels):
    # NB: don't use the dt/length interval dtype, integer types are too small
    # to contain these huge chunk-wide intervals
    return [
        (('Lowest start time observed in the chunk', 'time'), np.int64),
        (('Highest endt ime observed in the chunk', 'endtime'), np.int64),
        (('Number of pulses', 'pulse_count'),
         (np.int64, n_channels)),
        (('Number of lone pulses', 'lone_pulse_count'),
         (np.int64, n_channels)),
        (('Integral of all pulses in ADC_count x samples', 'pulse_area'),
         (np.int64, n_channels)),
        (('Integral of lone pulses in ADC_count x samples', 'lone_pulse_area'),
         (np.int64, n_channels)),
    ]


def count_pulses(records, n_channels):
    """Return array with one element, with pulse count info from records"""
    if len(records):
        result = np.zeros(1, dtype=pulse_count_dtype(n_channels))
        _count_pulses(records, n_channels, result)
        return result
    return np.zeros(0, dtype=pulse_count_dtype(n_channels))


@numba.njit(cache=True, nogil=True)
def _count_pulses(records, n_channels, result):
    count = np.zeros(n_channels, dtype=np.int64)
    lone_count = np.zeros(n_channels, dtype=np.int64)
    area = np.zeros(n_channels, dtype=np.int64)
    lone_area = np.zeros(n_channels, dtype=np.int64)

    last_end_seen = 0
    next_start = 0

    # Array of booleans to track whether we are currently in a lone pulse
    # in each channel
    in_lone_pulse = np.zeros(n_channels, dtype=np.bool_)

    for r_i, r in enumerate(records):
        if r_i != len(records) - 1:
            next_start = records[r_i + 1]['time']

        ch = r['channel']
        if ch >= n_channels:
            print(ch)
            raise RuntimeError("Out of bounds channel in get_counts!")

        area[ch] += r['area']  # <-- Summing total area in channel

        if r['record_i'] == 0:
            count[ch] += 1

            if (r['time'] > last_end_seen
                    and r['time'] + r['pulse_length'] * r['dt'] < next_start):
                # This is a lone pulse
                lone_count[ch] += 1
                in_lone_pulse[ch] = True
                lone_area[ch] += r['area']
            else:
                in_lone_pulse[ch] = False

            last_end_seen = max(last_end_seen,
                                r['time'] + r['pulse_length'] * r['dt'])

        elif in_lone_pulse[ch]:
            # This is a subsequent fragment of a lone pulse
            lone_area[ch] += r['area']

    res = result[0]
    res['pulse_count'][:] = count[:]
    res['lone_pulse_count'][:] = lone_count[:]
    res['pulse_area'][:] = area[:]
    res['lone_pulse_area'][:] = lone_area[:]
    res['time'] = records[0]['time']
    res['endtime'] = last_end_seen


##
# Misc
##

@numba.njit(cache=True, nogil=True)
def _mask_and_not(x, mask):
    return x[mask], x[~mask]


@export
@numba.njit(cache=True, nogil=True)
def channel_split(rr, first_other_ch):
    """Return """
    return _mask_and_not(rr, rr['channel'] < first_other_ch)


@export
def check_overlaps(records, n_channels):
    """Raise a ValueError if any of the pulses in records overlap

    Assumes records is already sorted by time.
    """
    last_end = np.zeros(n_channels, dtype=np.int64)
    channel, time = _check_overlaps(records, last_end)
    if channel != -9999:
        raise ValueError(
            f"Bad data! In channel {channel}, a pulse starts at {time}, "
            f"BEFORE the previous pulse in that same channel ended "
            f"(at {last_end[channel]})")


@numba.njit(cache=True, nogil=True)
def _check_overlaps(records, last_end):
    for r in records:
        if r['time'] < last_end[r['channel']]:
            return r['channel'], r['time']
        last_end[r['channel']] = strax.endtime(r)
    return -9999, -9999
