from immutabledict import immutabledict
import numba
import numpy as np

import strax
import straxen

export, __all__ = strax.exporter()
__all__ += ['NO_PULSE_COUNTS']

# These are also needed in peaklets, since hitfinding is repeated
HITFINDER_OPTIONS = tuple([
    strax.Option(
        'hit_min_amplitude',
        default='pmt_commissioning_initial',
        help='Minimum hit amplitude in ADC counts above baseline. '
             'See straxen.hit_min_amplitude for options.'
    )])

HITFINDER_OPTIONS_he = tuple([
    strax.Option(
        'hit_min_amplitude_he', track=True,
        default="pmt_commissioning_initial_he",
        child_option=True, parent_option_name='hit_min_amplitude',
        help='Minimum hit amplitude in ADC counts above baseline for the high energy channels. '
             'See straxen.hit_min_amplitude for options.'
    )])

HE_PREAMBLE = """High energy channels: attenuated signals of the top PMT-array\n"""


@export
@strax.takes_config(
    strax.Option('hev_gain_model',
                 default=('disabled', None),
                 help='PMT gain model used in the software high-energy veto.'
                      'Specify as (model_type, model_config)'),
    strax.Option(
        'baseline_samples',
        default=40,
        help='Number of samples to use at the start of the pulse to determine '
             'the baseline'),
    # Tail veto options
    strax.Option(
        'tail_veto_threshold',
        default=0,
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
    strax.Option(
        'max_veto_value',
        default=None,
        help="Optionally pass a HE peak that exceeds this absolute area. "
             "(if performing a hard veto, can keep a few statistics.)"),

    # PMT pulse processing options
    strax.Option(
        'pmt_pulse_filter',
        default=None,
        help='Linear filter to apply to pulses, will be normalized.'),
    strax.Option(
        'save_outside_hits',
        default=(3, 20),
        help='Save (left, right) samples besides hits; cut the rest'),

    strax.Option(
        'n_tpc_pmts', type=int,
        help='Number of TPC PMTs'),

    strax.Option(
        'check_raw_record_overlaps',
        default=True, track=False,
        help='Crash if any of the pulses in raw_records overlap with others '
             'in the same channel'),
    strax.Option(
        'allow_sloppy_chunking',
        default=False, track=False,
        help=('Use a default baseline for incorrectly chunked fragments. '
              'This is a kludge for improperly converted XENON1T data.')),

    *HITFINDER_OPTIONS)
class PulseProcessing(strax.Plugin):
    """
    1. Split raw_records into:
     - (tpc) records
     - aqmon_records
     - pulse_counts

    For TPC records, apply basic processing:
        1. Flip, baseline, and integrate the waveform
        2. Apply software HE veto after high-energy peaks.
        3. Find hits, apply linear filter, and zero outside hits.
    
    pulse_counts holds some average information for the individual PMT
    channels for each chunk of raw_records. This includes e.g.
    number of recorded pulses, lone_pulses (pulses which do not
    overlap with any other pulse), or mean values of baseline and
    baseline rms channel.
    """
    __version__ = '0.2.3'

    parallel = 'process'
    rechunk_on_save = immutabledict(
        records=False,
        veto_regions=True,
        pulse_counts=True)
    compressor = 'lz4'

    depends_on = 'raw_records'

    provides = ('records', 'veto_regions', 'pulse_counts')
    data_kind = {k: k for k in provides}
    save_when = strax.SaveWhen.TARGET

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
        self.hev_enabled = (
                (self.config['hev_gain_model'][0] != 'disabled')
                and self.config['tail_veto_threshold'])
        if self.hev_enabled:
            self.to_pe = straxen.get_to_pe(self.run_id,
                                           self.config['hev_gain_model'],
                                           self.config['n_tpc_pmts'])
        
    def compute(self, raw_records, start, end):
        if self.config['check_raw_record_overlaps']:
            check_overlaps(raw_records, n_channels=3000)

        # Throw away any non-TPC records; this should only happen for XENON1T
        # converted data
        raw_records = raw_records[
            raw_records['channel'] < self.config['n_tpc_pmts']]

        # Convert everything to the records data type -- adds extra fields.
        r = strax.raw_to_records(raw_records)
        del raw_records

        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        # TODO: better to throw an error if something is nonzero
        strax.zero_out_of_bounds(r)

        strax.baseline(r,
                       baseline_samples=self.config['baseline_samples'],
                       allow_sloppy_chunking=self.config['allow_sloppy_chunking'],
                       flip=True)

        strax.integrate(r)

        pulse_counts = count_pulses(r, self.config['n_tpc_pmts'])
        pulse_counts['time'] = start
        pulse_counts['endtime'] = end

        if len(r) and self.hev_enabled:

            r, r_vetoed, veto_regions = software_he_veto(
                r, self.to_pe, end,
                area_threshold=self.config['tail_veto_threshold'],
                veto_length=self.config['tail_veto_duration'],
                veto_res=self.config['tail_veto_resolution'],
                pass_veto_extend=self.config['tail_veto_pass_extend'],
                pass_veto_fraction=self.config['tail_veto_pass_fraction'],
                max_veto_value=self.config['max_veto_value'])

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
                min_amplitude=straxen.hit_min_amplitude(
                    self.config['hit_min_amplitude']))

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
                    pulse_counts=pulse_counts,
                    veto_regions=veto_regions)

    
@export
@strax.takes_config(
    strax.Option('n_he_pmts', track=False, default=752,
                 help="Maximum channel of the he channels"),
    strax.Option('record_length', default=110, track=False, type=int,
                 help="Number of samples per raw_record"),
    *HITFINDER_OPTIONS_he)
class PulseProcessingHighEnergy(PulseProcessing):
    __doc__ = HE_PREAMBLE + PulseProcessing.__doc__
    __version__ = '0.0.1'
    provides = ('records_he', 'pulse_counts_he')
    data_kind = {k: k for k in provides}
    rechunk_on_save = immutabledict(
        records_he=False,
        pulse_counts_he=True)
    depends_on = 'raw_records_he'
    compressor = 'lz4'
    child_plugin = True
    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        dtype = dict()
        dtype['records_he'] = strax.record_dtype(self.config["record_length"])
        dtype['pulse_counts_he'] = pulse_count_dtype(self.config['n_he_pmts'])
        return dtype

    def setup(self):
        self.hev_enabled = False
        self.config['n_tpc_pmts'] = self.config['n_he_pmts']
        self.config['hit_min_amplitude'] = self.config['hit_min_amplitude_he']

    def compute(self, raw_records_he, start, end):
        result = super().compute(raw_records_he, start, end)
        return dict(records_he=result['records'],
                    pulse_counts_he=result['pulse_counts'])

##
# Software HE Veto
##


@export
def software_he_veto(records, to_pe, chunk_end,
                     area_threshold=int(1e5),
                     veto_length=int(3e6),
                     veto_res=int(1e3),
                     pass_veto_fraction=0.01,
                     pass_veto_extend=3,
                     max_veto_value=None):
    """Veto veto_length (time in ns) after peaks larger than
    area_threshold (in PE).

    Further large peaks inside the veto regions are still passed:
    We sum the waveform inside the veto region (with time resolution
    veto_res in ns) and pass regions within pass_veto_extend samples
    of samples with amplitude above pass_veto_fraction times the maximum.

    :returns: (preserved records, vetoed records, veto intervals).

    :param records: PMT records
    :param to_pe: ADC to PE conversion factors for the channels in records.
    :param chunk_end: Endtime of chunk to set as maximum ceiling for the veto period
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
    :param max_veto_value: if not None, pass peaks that exceed this area
    no matter what.
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
    veto_start = peaks['time']
    veto_end = np.clip(peaks['time'] + veto_length,
                       None,
                       chunk_end)
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
    if max_veto_value is not None:
        pass_veto = strax.find_hits(regions, min_amplitude=max_veto_value)
    else:
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
    return tuple(list(mask_and_not(records, veto_mask)) + [veto])

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
        (('Start time of the chunk', 'time'), np.int64),
        (('End time of the chunk', 'endtime'), np.int64),
        (('Number of pulses', 'pulse_count'),
         (np.int64, n_channels)),
        (('Number of lone pulses', 'lone_pulse_count'),
         (np.int64, n_channels)),
        (('Integral of all pulses in ADC_count x samples', 'pulse_area'),
         (np.int64, n_channels)),
        (('Integral of lone pulses in ADC_count x samples', 'lone_pulse_area'),
         (np.int64, n_channels)),
        (('Average baseline', 'baseline_mean'),
         (np.int16, n_channels)),
        (('Average baseline rms', 'baseline_rms_mean'),
         (np.float32, n_channels)),
    ]


def count_pulses(records, n_channels):
    """Return array with one element, with pulse count info from records"""
    if len(records):
        result = np.zeros(1, dtype=pulse_count_dtype(n_channels))
        _count_pulses(records, n_channels, result)
        return result
    return np.zeros(0, dtype=pulse_count_dtype(n_channels))


NO_PULSE_COUNTS = -9999  # Special value required by average_baseline in case counts = 0
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
    baseline_buffer = np.zeros(n_channels, dtype=np.float64)
    baseline_rms_buffer = np.zeros(n_channels, dtype=np.float64)
    for r_i, r in enumerate(records):
        if r_i != len(records) - 1:
            next_start = records[r_i + 1]['time']

        ch = r['channel']
        if ch >= n_channels:
            print('Channel:', ch)
            raise RuntimeError("Out of bounds channel in get_counts!")

        area[ch] += r['area']  # <-- Summing total area in channel

        if r['record_i'] == 0:
            count[ch] += 1
            baseline_buffer[ch] += r['baseline']
            baseline_rms_buffer[ch] += r['baseline_rms'] 

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
    means = (baseline_buffer/count)
    means[np.isnan(means)] = NO_PULSE_COUNTS
    res['baseline_mean'][:] = means[:]
    res['baseline_rms_mean'][:] = (baseline_rms_buffer/count)[:]


##
# Misc
##
@export
@numba.njit(cache=True, nogil=True)
def mask_and_not(x, mask):
    return x[mask], x[~mask]


@export
@numba.njit(cache=True, nogil=True)
def channel_split(rr, first_other_ch):
    """Return """
    return mask_and_not(rr, rr['channel'] < first_other_ch)


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
