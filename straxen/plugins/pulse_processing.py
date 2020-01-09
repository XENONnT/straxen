import numba
import numpy as np

import strax
from straxen import get_to_pe
export, __all__ = strax.exporter()

# Number of TPC PMTs. Hardcoded for now...
n_tpc = 248


@export
@strax.takes_config(
    strax.Option(
        'to_pe_file',
        default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',    # noqa
        help='URL of the to_pe conversion factors'),

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
        'hit_threshold',
        default=15,
        help='Hitfinder threshold in ADC counts above baseline')
)
class PulseProcessing(strax.Plugin):
    """
    1. Split raw_records into:
     - tpc_records
     - diagnostic_records
     - aqmon_records
    Perhaps this should be done by DAQreader in the future

    For TPC records, apply basic processing:

    2. Apply software HE veto after high-energy peaks.
    3. Find hits, apply linear filter, and zero outside hits.
    """
    __version__ = '0.0.3'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'raw_records'

    provides = ('records', 'diagnostic_records', 'aqmon_records',
                'veto_regions', 'pulse_counts')
    data_kind = {k: k for k in provides}

    def infer_dtype(self):
        # Get record_length from the plugin making raw_records
        rr_dtype = self.deps['raw_records'].dtype_for('raw_records')
        record_length = len(np.zeros(1, rr_dtype)[0]['data'])

        dtype = dict()
        for p in self.provides:
            if p.endswith('records'):
                dtype[p] = strax.record_dtype(record_length)

        dtype['veto_regions'] = strax.hit_dtype
        dtype['pulse_counts'] = pulse_count_dtype(n_tpc)

        return dtype

    def setup(self):
        self.to_pe = get_to_pe(self.run_id, self.config['to_pe_file'])

    def compute(self, raw_records):
        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        strax.zero_out_of_bounds(raw_records)

        ##
        # Split off non-TPC records and count TPC pulses
        # (perhaps we should migrate this to DAQRreader in the future)
        ##
        r, other = channel_split(raw_records, n_tpc)
        pulse_counts = count_pulses(r, n_tpc)
        diagnostic_records, aqmon_records = channel_split(other, 254)

        ##
        # Process the TPC records
        ##
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
            hits = strax.find_hits(r, threshold=self.config['hit_threshold'])

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

    # 2. Find initial veto regions around these peaks
    # (with a generous right extension)
    veto_start, veto_end = strax.find_peak_groups(
        peaks,
        gap_threshold=veto_length + 2 * veto_res,
        right_extension=veto_length,
        left_extension=veto_res)
    veto_end = veto_end.clip(0, strax.endtime(records[-1]))
    veto_length = veto_end - veto_start
    # dtype is like record (since we want to use hitfiding etc)
    # but with float32 waveform
    regions = np.zeros(
        len(veto_start),
        dtype=strax.interval_dtype + [
            ("data", (np.float32, veto_n)),
            ("baseline", np.float32),
            ("reduction_level", np.int64),
            ("record_i", np.int64),
            ("pulse_length", np.int64),
        ])
    regions['time'] = veto_start
    regions['length'] = veto_length
    regions['pulse_length'] = veto_length
    regions['dt'] = veto_res

    if not len(regions):
        # No veto anywhere in this data
        return records, records[:0], np.zeros(0, strax.hit_dtype)

    # 3. Find pass_veto regios with big peaks inside the veto regions.
    # For this we compute a rough sum waveform (at low resolution,
    # without looping over the pulse data)
    rough_sum(regions, records, to_pe, veto_n, veto_res)
    regions['data'] /= np.max(regions['data'], axis=1)[:, np.newaxis]
    pass_veto = strax.find_hits(regions, threshold=pass_veto_fraction)

    # 4. Extend these by a few samples and inverse to find veto regions
    regions['data'] = 1
    regions = strax.cut_outside_hits(
        regions,
        pass_veto,
        left_extension=pass_veto_extend,
        right_extension=pass_veto_extend)
    regions['data'] = 1 - regions['data']
    veto = strax.find_hits(regions, threshold=0.5)
    # Do not remove very tiny regions
    veto = veto[veto['length'] > 2 * pass_veto_extend]

    # 5. Apply the veto and return results
    veto_mask = strax.fully_contained_in(records, veto) == -1
    return tuple(list(_mask_and_not(records, veto_mask)) + [veto])


@numba.njit
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
    result = np.zeros(1, dtype=pulse_count_dtype(n_channels))
    if len(records):
        _count_pulses(records, n_channels, result)
    return result


@numba.njit
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

@numba.njit
def _mask_and_not(x, mask):
    return x[mask], x[~mask]


@export
@numba.njit
def channel_split(rr, first_other_ch):
    """Return """
    return _mask_and_not(rr, rr['channel'] < first_other_ch)