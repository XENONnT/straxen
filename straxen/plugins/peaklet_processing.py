import numba
import numpy as np
import strax
from immutabledict import immutabledict
from strax.processing.general import _touching_windows
import straxen
from .pulse_processing import HE_PREAMBLE


export, __all__ = strax.exporter()
FAKE_MERGED_S2_TYPE = -42


@export
class Peaklets(strax.Plugin):
    """
    Split records into:
     - peaklets
     - lone_hits

    Peaklets are very aggressively split peaks such that we are able
    to find S1-S2s even if they are close to each other. (S2) Peaks
    that are split into too many peaklets will be merged later on.

    To get Peaklets from records apply/do:
        1. Hit finding
        2. Peak finding
        3. Peak splitting using the natural breaks algorithm
        4. Compute the digital sum waveform

    Lone hits are all hits which are outside of any peak. The area of
    lone_hits includes the left and right hit extension, except the
    extension overlaps with any peaks or other hits.
    """
    depends_on = ('records',)
    provides = ('peaklets', 'lone_hits')
    data_kind = dict(peaklets='peaklets',
                     lone_hits='lone_hits')
    parallel = 'process'
    compressor = 'zstd'

    __version__ = '0.6.0'

    peaklet_gap_threshold = straxen.URLConfig(default=700, infer_type=False,
                 help="No hits for this many ns triggers a new peak")

    peak_left_extension = straxen.URLConfig(default=30, infer_type=False,
                 help="Include this many ns left of hits in peaks")

    peak_right_extension = straxen.URLConfig(default=200, infer_type=False,
                 help="Include this many ns right of hits in peaks")

    peak_min_pmts = straxen.URLConfig(default=2, infer_type=False,
                 help="Minimum number of contributing PMTs needed to define a peak")

    peak_split_gof_threshold = straxen.URLConfig(
                 # See https://xe1t-wiki.lngs.infn.it/doku.php?id=
                 # xenon:xenonnt:analysis:strax_clustering_classification
                 # #natural_breaks_splitting
                 # for more information
                 default=(
                     None,  # Reserved
                     ((0.5, 1.0), (6.0, 0.4)),
                     ((2.5, 1.0), (5.625, 0.4))), infer_type=False,
                 help='Natural breaks goodness of fit/split threshold to split '
                      'a peak. Specify as tuples of (log10(area), threshold).')

    peak_split_filter_wing_width = straxen.URLConfig(default=70, infer_type=False,
                 help='Wing width of moving average filter for '
                      'low-split natural breaks')

    peak_split_min_area = straxen.URLConfig(default=40., infer_type=False,
                 help='Minimum area to evaluate natural breaks criterion. '
                      'Smaller peaks are not split.')
                      
    peak_split_iterations = straxen.URLConfig(default=20, infer_type=False,
                 help='Maximum number of recursive peak splits to do.')
                 
    diagnose_sorting = straxen.URLConfig(track=False, default=False, infer_type=False,
                 help="Enable runtime checks for sorting and disjointness")
                 
    gain_model = straxen.URLConfig(infer_type=False,
                 help='PMT gain model. Specify as URL or explicit value'
                 )
                 
    tight_coincidence_window_left = straxen.URLConfig(default=50, infer_type=False,
                 help="Time range left of peak center to call "
                      "a hit a tight coincidence (ns)")
                      
    tight_coincidence_window_right = straxen.URLConfig(default=50, infer_type=False,
                 help="Time range right of peak center to call "
                      "a hit a tight coincidence (ns)")
                      
    n_tpc_pmts = straxen.URLConfig(type=int,
                 help='Number of TPC PMTs')
                 
    saturation_correction_on = straxen.URLConfig(default=True, infer_type=False,
                 help='On off switch for saturation correction')
                 
    saturation_reference_length = straxen.URLConfig(default=100, infer_type=False,
                 help="Maximum number of reference sample used "
                      "to correct saturated samples")
                      
    saturation_min_reference_length = straxen.URLConfig(default=20, infer_type=False,
                 help="Minimum number of reference sample used "
                      "to correct saturated samples")
                      
    peaklet_max_duration = straxen.URLConfig(default=int(10e6), infer_type=False,
                 help="Maximum duration [ns] of a peaklet")
                 
    channel_map = straxen.URLConfig(track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number.")
                      
    hit_min_amplitude = straxen.URLConfig(
        track=True, infer_type=False,
        default='cmt://hit_thresholds_tpc?version=ONLINE&run_id=plugin.run_id',
        help='Minimum hit amplitude in ADC counts above baseline. '
                'Specify as a tuple of length n_tpc_pmts, or a number,'
                'or a string like "pmt_commissioning_initial" which means calling'
                'hitfinder_thresholds.py'
                'or a tuple like (correction=str, version=str, nT=boolean),'
                'which means we are using cmt.'
        )

    def infer_dtype(self):
        return dict(peaklets=strax.peak_dtype(
                        n_channels=self.n_tpc_pmts),
                    lone_hits=strax.hit_dtype)

    def setup(self):
        if self.peak_min_pmts > 2:
            # Can fix by re-splitting,
            raise NotImplementedError(
                f"Raising the peak_min_pmts to {self.peak_min_pmts} "
                f"interferes with lone_hit definition. "
                f"See github.com/XENONnT/straxen/issues/295")

        self.to_pe = self.gain_model

        self.hit_thresholds = self.hit_min_amplitude
            
        self.channel_range = self.channel_map['tpc']

    def compute(self, records, start, end):
        r = records

        hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)

        # Remove hits in zero-gain channels
        # they should not affect the clustering!
        hits = hits[self.to_pe[hits['channel']] != 0]

        hits = strax.sort_by_time(hits)

        # Use peaklet gap threshold for initial clustering
        # based on gaps between hits
        peaklets = strax.find_peaks(
            hits, self.to_pe,
            gap_threshold=self.peaklet_gap_threshold,
            left_extension=self.peak_left_extension,
            right_extension=self.peak_right_extension,
            min_channels=self.peak_min_pmts,
            result_dtype=self.dtype_for('peaklets'),
            max_duration=self.peaklet_max_duration,
        )

        # Make sure peaklets don't extend out of the chunk boundary
        # This should be very rare in normal data due to the ADC pretrigger
        # window.
        self.clip_peaklet_times(peaklets, start, end)

        # Get hits outside peaklets, and store them separately.
        # fully_contained is OK provided gap_threshold > extension,
        # which is asserted inside strax.find_peaks.
        is_lone_hit = strax.fully_contained_in(hits, peaklets) == -1
        lone_hits = hits[is_lone_hit]
        strax.integrate_lone_hits(
            lone_hits, records, peaklets,
            save_outside_hits=(self.peak_left_extension,
                               self.peak_right_extension),
            n_channels=len(self.to_pe))

        # Compute basic peak properties -- needed before natural breaks
        hits = hits[~is_lone_hit]
        # Define regions outside of peaks such that _find_hit_integration_bounds
        # is not extended beyond a peak.
        outside_peaks = self.create_outside_peaks_region(peaklets, start, end)
        strax.find_hit_integration_bounds(
            hits, outside_peaks, records,
            save_outside_hits=(self.peak_left_extension,
                               self.peak_right_extension),
            n_channels=len(self.to_pe),
            allow_bounds_beyond_records=True,
        )

        # Transform hits to hitlets for naming conventions. A hit refers
        # to the central part above threshold a hitlet to the entire signal
        # including the left and right extension.
        # (We are not going to use the actual hitlet data_type here.)
        hitlets = hits
        del hits

        hitlet_time_shift = (hitlets['left'] - hitlets['left_integration']) * hitlets['dt']
        hitlets['time'] = hitlets['time'] - hitlet_time_shift
        hitlets['length'] = (hitlets['right_integration'] - hitlets['left_integration'])
        hitlets = strax.sort_by_time(hitlets)
        rlinks = strax.record_links(records)

        strax.sum_waveform(peaklets, hitlets, r, rlinks, self.to_pe)

        strax.compute_widths(peaklets)

        # Split peaks using low-split natural breaks;
        # see https://github.com/XENONnT/straxen/pull/45
        # and https://github.com/AxFoundation/strax/pull/225
        peaklets = strax.split_peaks(
            peaklets, hitlets, r, rlinks, self.to_pe,
            algorithm='natural_breaks',
            threshold=self.natural_breaks_threshold,
            split_low=True,
            filter_wing_width=self.peak_split_filter_wing_width,
            min_area=self.peak_split_min_area,
            do_iterations=self.peak_split_iterations)

        # Saturation correction using non-saturated channels
        # similar method used in pax
        # see https://github.com/XENON1T/pax/pull/712
        # Cases when records is not writeable for unclear reason
        # only see this when loading 1T test data
        # more details on https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
        if not r['data'].flags.writeable:
            r = r.copy()

        if self.saturation_correction_on:
            peak_list = peak_saturation_correction(
                r, rlinks, peaklets, hitlets, self.to_pe,
                reference_length=self.saturation_reference_length,
                min_reference_length=self.saturation_min_reference_length)

            # Compute the width again for corrected peaks
            strax.compute_widths(peaklets, select_peaks_indices=peak_list)

        # Compute tight coincidence level.
        # Making this a separate plugin would
        # (a) doing hitfinding yet again (or storing hits)
        # (b) increase strax memory usage / max_messages,
        #     possibly due to its currently primitive scheduling.
        hit_max_times = np.sort(
            hitlets['time']
            + hitlets['dt'] * hit_max_sample(records, hitlets)
            + hitlet_time_shift  # add time shift again to get correct maximum
        )
        peaklet_max_times = (
                peaklets['time']
                + np.argmax(peaklets['data'], axis=1) * peaklets['dt'])
        tight_coincidence_channel = get_tight_coin(
            hit_max_times,
            hitlets['channel'],
            peaklet_max_times,
            self.tight_coincidence_window_left,
            self.tight_coincidence_window_right,
            self.channel_range)

        peaklets['tight_coincidence'] = tight_coincidence_channel

        if self.diagnose_sorting and len(r):
            assert np.diff(r['time']).min(initial=1) >= 0, "Records not sorted"
            assert np.diff(hitlets['time']).min(initial=1) >= 0, "Hits/Hitlets not sorted"
            assert np.all(peaklets['time'][1:]
                          >= strax.endtime(peaklets)[:-1]), "Peaks not disjoint"

        # Update nhits of peaklets:
        counts = strax.touching_windows(hitlets, peaklets)
        counts = np.diff(counts, axis=1).flatten()
        peaklets['n_hits'] = counts

        return dict(peaklets=peaklets,
                    lone_hits=lone_hits)

    def natural_breaks_threshold(self, peaks):
        rise_time = -peaks['area_decile_from_midpoint'][:, 1]

        # This is ~1 for an clean S2, ~0 for a clean S1,
        # and transitions gradually in between.
        f_s2 = 8 * np.log10(rise_time.clip(1, 1e5) / 100)
        f_s2 = 1 / (1 + np.exp(-f_s2))

        log_area = np.log10(peaks['area'].clip(1, 1e7))
        thresholds = self.peak_split_gof_threshold
        return (
            f_s2 * np.interp(
                log_area,
                *np.transpose(thresholds[2]))
            + (1 - f_s2) * np.interp(
                log_area,
                *np.transpose(thresholds[1])))

    @staticmethod
    @numba.njit(nogil=True, cache=True)
    def clip_peaklet_times(peaklets, start, end):
        for p in peaklets:
            if p['time'] < start:
                p['time'] = start
            if strax.endtime(p) > end:
                p['length'] = (end - p['time']) // p['dt']

    @staticmethod
    def create_outside_peaks_region(peaklets, start, end):
        """
        Creates time intervals which are outside peaks.

        :param peaklets: Peaklets for which intervals should be computed.
        :param start: Chunk start
        :param end: Chunk end
        :return: array of strax.time_fields dtype.
        """
        if not len(peaklets):
            return np.zeros(0, dtype=strax.time_fields)
        
        outside_peaks = np.zeros(len(peaklets) + 1,
                                 dtype=strax.time_fields)
        
        outside_peaks[0]['time'] = start
        outside_peaks[0]['endtime'] = peaklets[0]['time']
        outside_peaks[1:-1]['time'] = strax.endtime(peaklets[:-1])
        outside_peaks[1:-1]['endtime'] = peaklets['time'][1:]
        outside_peaks[-1]['time'] = strax.endtime(peaklets[-1])
        outside_peaks[-1]['endtime'] = end
        return outside_peaks


@numba.jit(nopython=True, nogil=True, cache=True)
def peak_saturation_correction(records, rlinks, peaks, hitlets, to_pe,
                               reference_length=100,
                               min_reference_length=20,
                               use_classification=False,
                               ):
    """Correct the area and per pmt area of peaks from saturation
    :param records: Records
    :param rlinks: strax.record_links of corresponding records.
    :param peaks: Peaklets / Peaks
    :param hitlets: Hitlets found in records to build peaks.
        (Hitlets are hits including the left/right extension)
    :param to_pe: adc to PE conversion (length should equal number of PMTs)
    :param reference_length: Maximum number of reference sample used
    to correct saturated samples
    :param min_reference_length: Minimum number of reference sample used
    to correct saturated samples
    :param use_classification: Option of using classification to pick only S2
    """

    if not len(records):
        return
    if not len(peaks):
        return

    # Search for peaks with saturated channels
    mask = peaks['n_saturated_channels'] > 0
    if use_classification:
        mask &= peaks['type'] == 2
    peak_list = np.where(mask)[0]
    # Look up records that touch each peak
    record_ranges = _touching_windows(
        records['time'],
        strax.endtime(records),
        peaks[peak_list]['time'],
        strax.endtime(peaks[peak_list]))

    # Create temporary arrays for calculation
    dt = records[0]['dt']
    n_channels = len(peaks[0]['saturated_channel'])
    len_buffer = np.max(peaks['length'] * peaks['dt']) // dt + 1
    max_nrecord = len_buffer // len(records[0]['data']) + 1

    # Buff the sum wf [pe] of non-saturated channels
    b_sumwf = np.zeros(len_buffer, dtype=np.float32)
    # Buff the records 'data' [ADC] in saturated channels
    b_pulse = np.zeros((n_channels, len_buffer), dtype=np.int16)
    # Buff the corresponding record index of saturated channels
    b_index = np.zeros((n_channels, max_nrecord), dtype=np.int64)

    # Main
    for ix, peak_i in enumerate(peak_list):
        # reset buffers
        b_sumwf[:] = 0
        b_pulse[:] = 0
        b_index[:] = -1

        p = peaks[peak_i]
        channel_saturated = p['saturated_channel'] > 0

        for record_i in range(record_ranges[ix][0], record_ranges[ix][1]):
            r = records[record_i]
            r_slice, b_slice = strax.overlap_indices(
                r['time'] // dt, r['length'],
                p['time'] // dt, p['length'] * p['dt'] // dt)

            ch = r['channel']
            if channel_saturated[ch]:
                b_pulse[ch, slice(*b_slice)] += r['data'][slice(*r_slice)]
                b_index[ch, np.argmin(b_index[ch])] = record_i
            else:
                b_sumwf[slice(*b_slice)] += r['data'][slice(*r_slice)] \
                    * to_pe[ch]

        _peak_saturation_correction_inner(
            channel_saturated, records, p,
            to_pe, b_sumwf, b_pulse, b_index,
            reference_length, min_reference_length)

        # Back track sum wf downsampling
        peaks[peak_i]['length'] = p['length'] * p['dt'] / dt
        peaks[peak_i]['dt'] = dt

    strax.sum_waveform(peaks, hitlets, records, rlinks, to_pe, peak_list)
    return peak_list


@numba.jit(nopython=True, nogil=True, cache=True)
def _peak_saturation_correction_inner(channel_saturated, records, p,
                                      to_pe, b_sumwf, b_pulse, b_index,
                                      reference_length=100,
                                      min_reference_length=20,
                                      ):
    """Would add a third level loop in peak_saturation_correction
    Which is not ideal for numba, thus this function is written
    :param channel_saturated: (bool, n_channels)
    :param p: One peak/peaklet
    :param to_pe: adc to PE conversion (length should equal number of PMTs)
    :param b_sumwf, b_pulse, b_index: Filled buffers
    """
    dt = records['dt'][0]
    n_channels = len(channel_saturated)

    for ch in range(n_channels):
        if not channel_saturated[ch]:
            continue
        b = b_pulse[ch]
        r0 = records[b_index[ch][0]]

        # Define the reference region as reference_length before the first saturation point
        # unless there are not enough samples
        bl = np.inf
        for record_i in b_index[ch]:
            if record_i == -1:
                break
            bl = min(bl, records['baseline'][record_i])

        s0 = np.argmax(b >= np.int16(bl))
        ref = slice(max(0, s0-reference_length), s0)

        if (b[ref] * to_pe[ch] > 1).sum() < min_reference_length:
            # the pulse is saturated, but there are not enough reference samples to get a good ratio
            # This actually distinguished between S1 and S2 and will only correct S2 signals
            continue
        if (b_sumwf[ref] > 1).sum() < min_reference_length:
            # the same condition applies to the waveform model
            continue
        if np.sum(b[ref]) * to_pe[ch] / np.sum(b_sumwf[ref]) > 1:
            # The pulse is saturated, but insufficient information is available in the other channels
            # to reliably reconstruct it
            continue

        scale = np.sum(b[ref]) / np.sum(b_sumwf[ref])

        # Loop over the record indices of the saturated channel (saved in b_index buffer)
        for record_i in b_index[ch]:
            if record_i == -1:
                break
            r = records[record_i]
            r_slice, b_slice = strax.overlap_indices(
                r['time'] // dt, r['length'],
                p['time'] // dt + s0,  p['length'] * p['dt'] // dt - s0)

            if r_slice[1] == r_slice[0]:  # This record proceeds saturation
                continue
            b_slice = b_slice[0] + s0, b_slice[1] + s0

            # First is finding the highest point in the desaturated record
            # because we need to bit shift the whole record if it exceeds int16 range
            apax = scale * max(b_sumwf[slice(*b_slice)])

            if np.int32(apax) >= 2**15:  # int16(2**15) is -2**15
                bshift = int(np.floor(np.log2(apax) - 14))

                tmp = r['data'].astype(np.int32)
                tmp[slice(*r_slice)] = b_sumwf[slice(*b_slice)] * scale

                r['area'] = np.sum(tmp)  # Auto covert to int64
                r['data'][:] = np.right_shift(tmp, bshift)
                r['amplitude_bit_shift'] += bshift
            else:
                r['data'][slice(*r_slice)] = b_sumwf[slice(*b_slice)] * scale
                r['area'] = np.sum(r['data'])


@export
class PeakletsHighEnergy(Peaklets):
    __doc__ = HE_PREAMBLE + Peaklets.__doc__
    depends_on = 'records_he'
    provides = 'peaklets_he'
    data_kind = 'peaklets_he'
    __version__ = '0.0.2'
    child_plugin = True
    save_when = strax.SaveWhen.TARGET

    n_he_pmts = straxen.URLConfig(track=False, default=752, infer_type=False,
                 help="Maximum channel of the he channels")
                 
    he_channel_offset = straxen.URLConfig(track=False, default=500, infer_type=False,
                 help="Minimum channel number of the he channels")
                 
    le_to_he_amplification = straxen.URLConfig(default=20, track=True, infer_type=False,
                 help="Difference in amplification between low energy and high "
                      "energy channels")
                      
    peak_min_pmts_he = straxen.URLConfig(default=2, infer_type=False,
                 child_option=True, parent_option_name='peak_min_pmts',
                 track=True,
                 help="Minimum number of contributing PMTs needed to define a peak")
                 
    saturation_correction_on_he = straxen.URLConfig(default=False, infer_type=False,
                 child_option=True, parent_option_name='saturation_correction_on',
                 track=True,
                 help='On off switch for saturation correction for High Energy'
                      ' channels')

    hit_min_amplitude_he = straxen.URLConfig(
        default='cmt://hit_thresholds_he?version=ONLINE&run_id=plugin.run_id', track=True, infer_type=False,
        help='Minimum hit amplitude in ADC counts above baseline. '
                'Specify as a tuple of length n_tpc_pmts, or a number,'
                'or a string like "pmt_commissioning_initial" which means calling'
                'hitfinder_thresholds.py'
                'or a tuple like (correction=str, version=str, nT=boolean),'
                'which means we are using cmt.'
        )

    def infer_dtype(self):
        return strax.peak_dtype(n_channels=self.n_he_pmts)

    def setup(self):
        self.to_pe = self.gain_model                 
        buffer_pmts = np.zeros(self.he_channel_offset)
        self.to_pe = np.concatenate((buffer_pmts, self.to_pe))
        self.to_pe *= self.le_to_he_amplification

        self.hit_thresholds = self.hit_min_amplitude_he

        self.channel_range = self.channel_map['he']

    def compute(self, records_he, start, end):
        result = super().compute(records_he, start, end)
        return result['peaklets']


@export
class PeakletClassification(strax.Plugin):
    """Classify peaklets as unknown, S1, or S2."""
    __version__ = '3.0.3'

    provides = 'peaklet_classification'
    depends_on = ('peaklets',)
    parallel = True
    dtype = (strax.peak_interval_dtype
             + [('type', np.int8, 'Classification of the peak(let)')])

    s1_risetime_area_parameters = straxen.URLConfig(default=(50, 80, 12), type=(list, tuple),
                 help="norm, const, tau in the empirical boundary in the risetime-area plot")

    s1_risetime_aft_parameters = straxen.URLConfig(default=(-1, 2.6), type=(list, tuple),
                 help=("Slope and offset in exponential of emperical boundary in the rise time-AFT "
                      "plot. Specified as (slope, offset)"))
                      
    s1_flatten_threshold_aft = straxen.URLConfig(default=(0.6, 100), type=(tuple, list),
                 help=("Threshold for AFT, above which we use a flatted boundary for rise time" 
                       "Specified values: (AFT boundary, constant rise time)."))
                       
    n_top_pmts = straxen.URLConfig(default=straxen.n_top_pmts, type=int,
                 help="Number of top PMTs")
                 
    s1_max_rise_time_post100 = straxen.URLConfig(default=200, type=(int, float),
                 help="Maximum S1 rise time for > 100 PE [ns]")
                 
    s1_min_coincidence = straxen.URLConfig(default=2, type=int,
                 help="Minimum tight coincidence necessary to make an S1")
                 
    s2_min_pmts = straxen.URLConfig(default=4, type=int,
                 help="Minimum number of PMTs contributing to an S2")

    @staticmethod
    def upper_rise_time_area_boundary(area, norm, const, tau):
        """
        Function which determines the upper boundary for the rise-time
        for a given area.
        """
        return norm*np.exp(-area/tau) + const

    @staticmethod
    def upper_rise_time_aft_boundary(aft, slope, offset, aft_boundary, flat_threshold):
        """
        Function which computes the upper rise time boundary as a function
        of area fraction top.
        """
        res = 10**(slope * aft + offset)
        res[aft >= aft_boundary] = flat_threshold
        return res

    def compute(self, peaklets):
        ptype = np.zeros(len(peaklets), dtype=np.int8)

        # Properties needed for classification:
        rise_time = -peaklets['area_decile_from_midpoint'][:, 1]
        n_channels = (peaklets['area_per_channel'] > 0).sum(axis=1)
        n_top = self.n_top_pmts
        area_top = peaklets['area_per_channel'][:, :n_top].sum(axis=1)
        area_total = peaklets['area_per_channel'].sum(axis=1)
        area_fraction_top = area_top/area_total

        is_large_s1 = (peaklets['area'] >= 100)
        is_large_s1 &= (rise_time <= self.s1_max_rise_time_post100)
        is_large_s1 &= peaklets['tight_coincidence'] >= self.s1_min_coincidence

        is_small_s1 = peaklets["area"] < 100
        is_small_s1 &= rise_time < self.upper_rise_time_area_boundary(
            peaklets["area"],
            *self.s1_risetime_area_parameters,
        )

        is_small_s1 &= rise_time < self.upper_rise_time_aft_boundary(
            area_fraction_top,
            *self.s1_risetime_aft_parameters,
            *self.s1_flatten_threshold_aft,
        )

        is_small_s1 &= peaklets['tight_coincidence'] >= self.s1_min_coincidence

        ptype[is_large_s1 | is_small_s1] = 1

        is_s2 = n_channels >= self.s2_min_pmts
        is_s2[is_large_s1 | is_small_s1] = False
        ptype[is_s2] = 2

        return dict(type=ptype,
                    time=peaklets['time'],
                    dt=peaklets['dt'],
                    # Channel is added so the field order of the merger of
                    # peaklet_classification and peaklets matches that
                    # of peaklets.
                    # This way S2 merging works on arrays of the same dtype.
                    channel=-1,
                    length=peaklets['length'])


@export
class PeakletClassificationHighEnergy(PeakletClassification):
    __doc__ = HE_PREAMBLE + PeakletClassification.__doc__
    provides = 'peaklet_classification_he'
    depends_on = ('peaklets_he',)
    __version__ = '0.0.2'
    child_plugin = True

    def compute(self, peaklets_he):
        return super().compute(peaklets_he)


@export
class MergedS2s(strax.OverlapWindowPlugin):
    """
    Merge together peaklets if peak finding favours that they would
    form a single peak instead.
    """
    __version__ = '0.5.0'

    depends_on = ('peaklets', 'peaklet_classification', 'lone_hits')
    data_kind = 'merged_s2s'
    provides = 'merged_s2s'
    
    s2_merge_max_duration = straxen.URLConfig(default=50_000, infer_type=False,
                 help="Do not merge peaklets at all if the result would be a peak "
                      "longer than this [ns]")

    s2_merge_gap_thresholds = straxen.URLConfig(default=((1.7, 2.65e4), (4.0, 2.6e3), (5.0, 0.)),
                 infer_type=False,
                 help="Points to define maximum separation between peaklets to allow "
                      "merging [ns] depending on log10 area of the merged peak\n"
                      "where the gap size of the first point is the maximum gap to allow merging"
                      "and the area of the last point is the maximum area to allow merging. "
                      "The format is ((log10(area), max_gap), (..., ...), (..., ...))"
                 )

    gain_model = straxen.URLConfig(infer_type=False,
                 help='PMT gain model. Specify as '
                      '(str(model_config), str(version), nT-->boolean')

    merge_without_s1 = straxen.URLConfig(default=True, infer_type=False,
                 help="If true, S1s will be igored during the merging. "
                      "It's now possible for a S1 to be inside a S2 post merging")

    def setup(self):
        self.to_pe = self.gain_model

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps['peaklets'].dtype_for('peaklets'))

    def get_window_size(self):
        return 5 * (int(self.s2_merge_gap_thresholds[0][1])
                    + self.s2_merge_max_duration)

    def compute(self, peaklets, lone_hits):
        if self.merge_without_s1:
            peaklets = peaklets[peaklets['type'] != 1]

        if len(peaklets) <= 1:
            return np.zeros(0, dtype=self.dtype)

        gap_thresholds = self.s2_merge_gap_thresholds
        max_gap = gap_thresholds[0][1]
        max_area = 10 ** gap_thresholds[-1][0]

        if max_gap < 0:
            # Do not merge at all
            return np.zeros(0, dtype=self.dtype)
        else:
            # Max gap and area should be set by the gap thresholds
            # to avoid contradictions
            start_merge_at, end_merge_at = self.get_merge_instructions(
                peaklets['time'], strax.endtime(peaklets),
                areas=peaklets['area'],
                types=peaklets['type'],
                gap_thresholds=gap_thresholds,
                max_duration=self.s2_merge_max_duration,
                max_gap=max_gap,
                max_area=max_area,
            )
            merged_s2s = strax.merge_peaks(
                peaklets,
                start_merge_at, end_merge_at,
                max_buffer=int(self.s2_merge_max_duration//np.gcd.reduce(peaklets['dt'])),
            )
            merged_s2s['type'] = 2
            
            # Updated time and length of lone_hits and sort again:
            lh = np.copy(lone_hits)
            del lone_hits
            lh_time_shift = (lh['left'] - lh['left_integration']) *lh['dt']
            lh['time'] = lh['time'] - lh_time_shift
            lh['length'] = (lh['right_integration'] - lh['left_integration'])
            lh = strax.sort_by_time(lh)
            strax.add_lone_hits(merged_s2s, lh, self.to_pe)

            strax.compute_widths(merged_s2s)

        return merged_s2s

    @staticmethod
    @numba.njit(cache=True, nogil=True)
    def get_merge_instructions(
            peaklet_starts, peaklet_ends, areas, types,
            gap_thresholds, max_duration, max_gap, max_area):
        """
        Finding the group of peaklets to merge. To do this start with the
        smallest gaps and keep merging until the new, merged S2 has such a
        large area or gap to adjacent peaks that merging is not required
        anymore.
        see https://github.com/XENONnT/straxen/pull/548 and https://github.com/XENONnT/straxen/pull/568

        :returns: list of the first index of peaklet to be merged and
        list of the exclusive last index of peaklet to be merged
        """

        peaklet_gaps = peaklet_starts[1:] - peaklet_ends[:-1]
        peaklet_start_index = np.arange(len(peaklet_starts))
        peaklet_end_index = np.arange(len(peaklet_starts))

        for gap_i in np.argsort(peaklet_gaps):
            start_idx = peaklet_start_index[gap_i]
            inclusive_end_idx = peaklet_end_index[gap_i + 1]
            sum_area = np.sum(areas[start_idx:inclusive_end_idx + 1])
            this_gap = peaklet_gaps[gap_i]

            if inclusive_end_idx < start_idx:
                raise ValueError('Something went wrong, left is bigger then right?!')

            if this_gap > max_gap:
                break
            if sum_area > max_area:
                # For very large S2s, we assume that natural breaks is taking care
                continue
            if (sum_area > 0) and (
                    this_gap > merge_s2_threshold(np.log10(sum_area),
                                                  gap_thresholds)):
                # The merged peak would be too large
                continue

            peak_duration = (peaklet_ends[inclusive_end_idx] - peaklet_starts[start_idx])
            if peak_duration >= max_duration:
                continue

            # Merge gap in other words this means p @ gap_i and p @gap_i + 1 share the same
            # start, end and area:
            peaklet_start_index[start_idx:inclusive_end_idx + 1] = peaklet_start_index[start_idx]
            peaklet_end_index[start_idx:inclusive_end_idx + 1] = peaklet_end_index[inclusive_end_idx]

        start_merge_at = np.unique(peaklet_start_index)
        end_merge_at = np.unique(peaklet_end_index)
        if not len(start_merge_at) == len(end_merge_at):
            raise ValueError('inconsistent start and end merge instructions')

        merge_start, merge_stop_exclusive = _filter_s1_starts(
            start_merge_at, types, end_merge_at)

        return merge_start, merge_stop_exclusive


@numba.njit(cache=True, nogil=True)
def _filter_s1_starts(start_merge_at, types, end_merge_at):
    for start_merge_idx, _ in enumerate(start_merge_at):
        while types[start_merge_at[start_merge_idx]] != 2:
            if end_merge_at[start_merge_idx] - start_merge_at[start_merge_idx] <= 1:
                break
            start_merge_at[start_merge_idx] += 1

    start_merge_with_s2 = types[start_merge_at] == 2
    merges_at_least_two_peaks = end_merge_at - start_merge_at >= 1

    keep_merges = start_merge_with_s2 & merges_at_least_two_peaks
    return start_merge_at[keep_merges], end_merge_at[keep_merges] + 1


@numba.njit(cache=True, nogil=True)
def merge_s2_threshold(log_area, gap_thresholds):
    """Return gap threshold for log_area of the merged S2
    with linear interpolation given the points in gap_thresholds
    :param log_area: Log 10 area of the merged S2
    :param gap_thresholds: tuple (n, 2) of fix points for interpolation
    """
    for i, (a1, g1) in enumerate(gap_thresholds):
        if log_area < a1:
            if i == 0:
                return g1
            a0, g0 = gap_thresholds[i - 1]
            return (log_area - a0) * (g1 - g0) / (a1 - a0) + g0
    return gap_thresholds[-1][1]


@export
class MergedS2sHighEnergy(MergedS2s):
    __doc__ = HE_PREAMBLE + MergedS2s.__doc__
    depends_on = ('peaklets_he', 'peaklet_classification_he')
    data_kind = 'merged_s2s_he'
    provides = 'merged_s2s_he'
    __version__ = '0.0.1'
    child_plugin = True

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps['peaklets_he'].dtype_for('peaklets_he'))

    def compute(self, peaklets_he):
        # There are not any lone hits for the high energy channel, 
        #  so create a dummy for the compute method.
        lone_hits = np.zeros(0, dtype=strax.hit_dtype)
        return super().compute(peaklets_he, lone_hits)


@export
class Peaks(strax.Plugin):
    """
    Merge peaklets and merged S2s such that we obtain our peaks
    (replacing all peaklets that were later re-merged as S2s). As this
    step is computationally trivial, never save this plugin.
    """
    __version__ = '0.1.2'

    depends_on = ('peaklets', 'peaklet_classification', 'merged_s2s')
    data_kind = 'peaks'
    provides = 'peaks'
    parallel = True
    save_when = strax.SaveWhen.EXPLICIT

    
    diagnose_sorting = straxen.URLConfig( track=False, default=False, infer_type=False,
                 help="Enable runtime checks for sorting and disjointness")

    merge_without_s1 = straxen.URLConfig( default=True, infer_type=False,
                 help="If true, S1s will be igored during the merging. "
                      "It's now possible for a S1 to be inside a S2 post merging")

    def infer_dtype(self):
        return self.deps['peaklets'].dtype_for('peaklets')

    def compute(self, peaklets, merged_s2s):
        # Remove fake merged S2s from dirty hack, see above
        merged_s2s = merged_s2s[merged_s2s['type'] != FAKE_MERGED_S2_TYPE]
        
        if self.merge_without_s1:
            is_s1 = peaklets['type'] == 1
            peaks = strax.replace_merged(peaklets[~is_s1], merged_s2s)
            peaks = strax.sort_by_time(np.concatenate([peaklets[is_s1],
                                                       peaks]))
        else:
            peaks = strax.replace_merged(peaklets, merged_s2s)

        if self.diagnose_sorting:
            assert np.all(np.diff(peaks['time']) >= 0), "Peaks not sorted"
            if self.merge_without_s1:
                to_check = peaks['type'] != 1
            else:
                to_check = peaks['type'] != FAKE_MERGED_S2_TYPE

            assert np.all(peaks['time'][to_check][1:]
                            >= strax.endtime(peaks)[to_check][:-1]), "Peaks not disjoint"
        return peaks


@export
class PeaksHighEnergy(Peaks):
    __doc__ = HE_PREAMBLE + Peaks.__doc__
    depends_on = ('peaklets_he', 'peaklet_classification_he', 'merged_s2s_he')
    data_kind = 'peaks_he'
    provides = 'peaks_he'
    __version__ = '0.0.1'
    child_ends_with = '_he'

    def infer_dtype(self):
        return self.deps['peaklets_he'].dtype_for('peaklets')

    def compute(self, peaklets_he, merged_s2s_he):
        return super().compute(peaklets_he, merged_s2s_he)


@numba.jit(nopython=True, nogil=True, cache=True)
def get_tight_coin(hit_max_times, hit_channel, peak_max_times, left, right,
                   channels=(0, 493)):
    """Calculates the tight coincidence based on PMT channels.

    Defined by number of hits within a specified time range of the
    the peak's maximum amplitude.
    Imitates tight_coincidence variable in pax:
    github.com/XENON1T/pax/blob/master/pax/plugins/peak_processing/BasicProperties.py

    :param hit_max_times: Time of the hit amplitude in ns.
    :param hit_channel: PMT channels of the hits
    :param peak_max_times: Time of the peaks maximum in ns.
    :param left: Left boundary in which we search for the tight
        coincidence in ns.
    :param right: Right boundary in which we search for the tight
        coincidence in ns.
    :param channel_range: (min/max) channel for the corresponding detector.

    :returns: n_coin_channel of length peaks containing the
        tight coincidence.
    """
    left_hit_i = 0
    n_coin_channel = np.zeros(len(peak_max_times), dtype=np.int16)
    start_ch, end_ch = channels
    channels_seen = np.zeros(end_ch-start_ch+1, dtype=np.bool_)

    # loop over peaks
    for p_i, p_t in enumerate(peak_max_times):
        channels_seen[:] = 0
        # loop over hits starting from the last one we left at
        for left_hit_i in range(left_hit_i, len(hit_max_times)):

            # if the hit is in the window, its a tight coin
            d = hit_max_times[left_hit_i] - p_t
            if (-left <= d) & (d <= right):
                channels_seen[hit_channel[left_hit_i]-start_ch] = 1

            # stop the loop when we know we're outside the range
            if d > right:
                n_coin_channel[p_i] = np.sum(channels_seen)
                break
        
        # Add channel information in case there are no hits beyond 
        # the last peak:
        n_coin_channel[p_i] = np.sum(channels_seen)

    return n_coin_channel


@numba.njit(cache=True, nogil=True)
def hit_max_sample(records, hits):
    """Return the index of the maximum sample for hits"""
    result = np.zeros(len(hits), dtype=np.int16)
    for i, h in enumerate(hits):
        r = records[h['record_i']]
        w = r['data'][h['left']:h['right']]
        result[i] = np.argmax(w)
    return result
