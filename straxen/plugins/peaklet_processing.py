import numba
import numpy as np
import strax
from immutabledict import immutabledict
from strax.processing.general import _touching_windows
import straxen
from .pulse_processing import HITFINDER_OPTIONS, HITFINDER_OPTIONS_he, HE_PREAMBLE
from straxen.get_corrections import is_cmt_option


export, __all__ = strax.exporter()
FAKE_MERGED_S2_TYPE = -42


@export
@strax.takes_config(
    strax.Option('peaklet_gap_threshold', default=700,
                 help="No hits for this many ns triggers a new peak"),
    strax.Option('peak_left_extension', default=30,
                 help="Include this many ns left of hits in peaks"),
    strax.Option('peak_right_extension', default=200,
                 help="Include this many ns right of hits in peaks"),
    strax.Option('peak_min_pmts', default=2,
                 help="Minimum number of contributing PMTs needed to define a peak"),
    strax.Option('peak_split_gof_threshold',
                 # See https://xe1t-wiki.lngs.infn.it/doku.php?id=
                 # xenon:xenonnt:analysis:strax_clustering_classification
                 # #natural_breaks_splitting
                 # for more information
                 default=(
                     None,  # Reserved
                     ((0.5, 1.0), (6.0, 0.4)),
                     ((2.5, 1.0), (5.625, 0.4))),
                 help='Natural breaks goodness of fit/split threshold to split '
                      'a peak. Specify as tuples of (log10(area), threshold).'),
    strax.Option('peak_split_filter_wing_width', default=70,
                 help='Wing width of moving average filter for '
                      'low-split natural breaks'),
    strax.Option('peak_split_min_area', default=40.,
                 help='Minimum area to evaluate natural breaks criterion. '
                      'Smaller peaks are not split.'),
    strax.Option('peak_split_iterations', default=20,
                 help='Maximum number of recursive peak splits to do.'),
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"),
    strax.Option('gain_model',
                 help='PMT gain model. Specify as '
                 '(str(model_config), str(version), nT-->boolean'),
    strax.Option('tight_coincidence_window_left', default=50,
                 help="Time range left of peak center to call "
                      "a hit a tight coincidence (ns)"),
    strax.Option('tight_coincidence_window_right', default=50,
                 help="Time range right of peak center to call "
                      "a hit a tight coincidence (ns)"),
    strax.Option('n_tpc_pmts', type=int,
                 help='Number of TPC PMTs'),
    strax.Option('saturation_correction_on', default=True,
                 help='On off switch for saturation correction'),
    strax.Option('saturation_reference_length', default=100,
                 help="Maximum number of reference sample used "
                      "to correct saturated samples"),
    strax.Option('saturation_min_reference_length', default=20,
                 help="Minimum number of reference sample used "
                      "to correct saturated samples"),
    strax.Option('peaklet_max_duration', default=int(10e6),
                 help="Maximum duration [ns] of a peaklet"),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number."),
    *HITFINDER_OPTIONS,
)
class Peaklets(strax.Plugin):
    """
    Split records into:
        -peaklets
        -lone_hits

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

    __version__ = '0.5.0'

    def infer_dtype(self):
        return dict(peaklets=strax.peak_dtype(
                        n_channels=self.config['n_tpc_pmts']),
                    lone_hits=strax.hit_dtype)

    def setup(self):
        if self.config['peak_min_pmts'] > 2:
            # Can fix by re-splitting,
            raise NotImplementedError(
                f"Raising the peak_min_pmts to {self.config['peak_min_pmts']} "
                f"interferes with lone_hit definition. "
                f"See github.com/XENONnT/straxen/issues/295")

        self.to_pe = straxen.get_correction_from_cmt(self.run_id,
                                       self.config['gain_model'])

        # Check config of `hit_min_amplitude` and define hit thresholds
        # if cmt config
        if is_cmt_option(self.config['hit_min_amplitude']):
            self.hit_thresholds = straxen.get_correction_from_cmt(self.run_id,
                self.config['hit_min_amplitude'])
        # if hitfinder_thresholds config
        elif isinstance(self.config['hit_min_amplitude'], str):
            self.hit_thresholds = straxen.hit_min_amplitude(
                self.config['hit_min_amplitude'])
        else: # int or array
            self.hit_thresholds = self.config['hit_min_amplitude']
            
        self.channel_range = self.config['channel_map']['tpc']

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
            gap_threshold=self.config['peaklet_gap_threshold'],
            left_extension=self.config['peak_left_extension'],
            right_extension=self.config['peak_right_extension'],
            min_channels=self.config['peak_min_pmts'],
            result_dtype=self.dtype_for('peaklets'),
            max_duration=self.config['peaklet_max_duration'],
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
            save_outside_hits=(self.config['peak_left_extension'],
                               self.config['peak_right_extension']),
            n_channels=len(self.to_pe))

        # Compute basic peak properties -- needed before natural breaks
        hits = hits[~is_lone_hit]
        # Define regions outside of peaks such that _find_hit_integration_bounds
        # is not extended beyond a peak.
        outside_peaks = self.create_outside_peaks_region(peaklets, start, end)
        strax.find_hit_integration_bounds(
            hits, outside_peaks, records,
            save_outside_hits=(self.config['peak_left_extension'],
                               self.config['peak_right_extension']),
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
            filter_wing_width=self.config['peak_split_filter_wing_width'],
            min_area=self.config['peak_split_min_area'],
            do_iterations=self.config['peak_split_iterations'])

        # Saturation correction using non-saturated channels
        # similar method used in pax
        # see https://github.com/XENON1T/pax/pull/712
        # Cases when records is not writeable for unclear reason
        # only see this when loading 1T test data
        # more details on https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
        if not r['data'].flags.writeable:
            r = r.copy()

        if self.config['saturation_correction_on']:
            peak_list = peak_saturation_correction(
                r, rlinks, peaklets, hitlets, self.to_pe,
                reference_length=self.config['saturation_reference_length'],
                min_reference_length=self.config['saturation_min_reference_length'])

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
        tight_coincidence, tight_coincidence_channel = get_tight_coin(
            hit_max_times,
            hitlets['channel'],
            peaklet_max_times,
            self.config['tight_coincidence_window_left'],
            self.config['tight_coincidence_window_right'],
            self.channel_range)

        peaklets['tight_coincidence'] = tight_coincidence
        peaklets['tight_coincidence_channel'] = tight_coincidence_channel

        if self.config['diagnose_sorting'] and len(r):
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
        thresholds = self.config['peak_split_gof_threshold']
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
@strax.takes_config(
    strax.Option('n_he_pmts', track=False, default=752,
                 help="Maximum channel of the he channels"),
    strax.Option('he_channel_offset', track=False, default=500,
                 help="Minimum channel number of the he channels"),
    strax.Option('le_to_he_amplification', default=20, track=True,
                 help="Difference in amplification between low energy and high "
                      "energy channels"),
    strax.Option('peak_min_pmts_he', default=2,
                 child_option=True, parent_option_name='peak_min_pmts',
                 track=True,
                 help="Minimum number of contributing PMTs needed to define a peak"),
    strax.Option('saturation_correction_on_he', default=False,
                 child_option=True, parent_option_name='saturation_correction_on',
                 track=True,
                 help='On off switch for saturation correction for High Energy'
                      ' channels'),
    *HITFINDER_OPTIONS_he
)
class PeakletsHighEnergy(Peaklets):
    __doc__ = HE_PREAMBLE + Peaklets.__doc__
    depends_on = 'records_he'
    provides = 'peaklets_he'
    data_kind = 'peaklets_he'
    __version__ = '0.0.2'
    child_plugin = True
    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        return strax.peak_dtype(n_channels=self.config['n_he_pmts'])

    def setup(self):
        self.to_pe = straxen.get_correction_from_cmt(self.run_id,
                                       self.config['gain_model'])

        buffer_pmts = np.zeros(self.config['he_channel_offset'])
        self.to_pe = np.concatenate((buffer_pmts, self.to_pe))
        self.to_pe *= self.config['le_to_he_amplification']

        # Check config of `hit_min_amplitude_he` and define hit thresholds
        # if cmt config
        if is_cmt_option(self.config['hit_min_amplitude_he']):
            self.hit_thresholds = straxen.get_correction_from_cmt(self.run_id,
                self.config['hit_min_amplitude_he'])
        # if hitfinder_thresholds config
        elif isinstance(self.config['hit_min_amplitude_he'], str):
            self.hit_thresholds = straxen.hit_min_amplitude(
                self.config['hit_min_amplitude_he'])
        else: # int or array
            self.hit_thresholds = self.config['hit_min_amplitude_he']
            
        self.channel_range = self.config['channel_map']['he']

    def compute(self, records_he, start, end):
        result = super().compute(records_he, start, end)
        return result['peaklets']


@export
@strax.takes_config(
    strax.Option('s1_max_rise_time', default=110,
                 help="Maximum S1 rise time for < 100 PE [ns]"),
    strax.Option('s1_max_rise_time_post100', default=200,
                 help="Maximum S1 rise time for > 100 PE [ns]"),
    strax.Option('s1_min_coincidence', default=2,
                 help="Minimum tight coincidence necessary to make an S1"),
    strax.Option('s2_min_pmts', default=4,
                 help="Minimum number of PMTs contributing to an S2"))
class PeakletClassification(strax.Plugin):
    """Classify peaklets as unknown, S1, or S2."""
    provides = 'peaklet_classification'
    depends_on = ('peaklets',)
    parallel = True
    dtype = (strax.peak_interval_dtype
             + [('type', np.int8, 'Classification of the peak(let)')])
    __version__ = '0.2.1'

    def compute(self, peaklets):
        peaks = peaklets

        ptype = np.zeros(len(peaklets), dtype=np.int8)

        # Properties needed for classification. Bit annoying these computations
        # are duplicated in peak_basics curently...
        rise_time = -peaks['area_decile_from_midpoint'][:, 1]
        n_channels = (peaks['area_per_channel'] > 0).sum(axis=1)

        is_s1 = (
           (rise_time <= self.config['s1_max_rise_time'])
            | ((rise_time <= self.config['s1_max_rise_time_post100'])
               & (peaks['area'] > 100)))
        is_s1 &= peaks['tight_coincidence'] >= self.config['s1_min_coincidence']
        ptype[is_s1] = 1

        is_s2 = n_channels >= self.config['s2_min_pmts']
        is_s2[is_s1] = False
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
@strax.takes_config(
    strax.Option('s2_merge_max_duration', default=50_000,
                 help="Do not merge peaklets at all if the result would be a peak "
                      "longer than this [ns]"),
    strax.Option('s2_merge_gap_thresholds', default=((1.7, 2.65e4), (4.0, 2.6e3), (5.0, 0.)),
                 help="Points to define maximum separation between peaklets to allow "
                      "merging [ns] depending on log10 area of the merged peak\n"
                      "where the gap size of the first point is the maximum gap to allow merging"
                      "and the area of the last point is the maximum area to allow merging. "
                      "The format is ((log10(area), max_gap), (..., ...), (..., ...))"
                 ),
    strax.Option('gain_model',
                 help='PMT gain model. Specify as '
                      '(str(model_config), str(version), nT-->boolean'),
    strax.Option('merge_without_s1', default=True,
                 help="If true, S1s will be igored during the merging. "
                      "It's now possible for a S1 to be inside a S2 post merging"),
)
class MergedS2s(strax.OverlapWindowPlugin):
    """
    Merge together peaklets if peak finding favours that they would
    form a single peak instead.
    """
    depends_on = ('peaklets', 'peaklet_classification', 'lone_hits')
    data_kind = 'merged_s2s'
    provides = 'merged_s2s'
    __version__ = '0.4.0'

    def setup(self):
        self.to_pe = straxen.get_correction_from_cmt(self.run_id,
                                                     self.config['gain_model'])

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps['peaklets'].dtype_for('peaklets'))

    def get_window_size(self):
        return 5 * (int(self.config['s2_merge_gap_thresholds'][0][1])
                    + self.config['s2_merge_max_duration'])

    def compute(self, peaklets, lone_hits):
        if self.config['merge_without_s1']:
            peaklets = peaklets[peaklets['type'] != 1]

        if len(peaklets) <= 1:
            return np.zeros(0, dtype=peaklets.dtype)

        gap_thresholds = self.config['s2_merge_gap_thresholds']
        max_gap = gap_thresholds[0][1]
        max_area = 10 ** gap_thresholds[-1][0]

        if max_gap < 0:
            # Do not merge at all
            return np.zeros(0, dtype=peaklets.dtype)
        else:
            # Max gap and area should be set by the gap thresholds
            # to avoid contradictions
            start_merge_at, end_merge_at = self.get_merge_instructions(
                peaklets['time'], strax.endtime(peaklets),
                areas=peaklets['area'],
                types=peaklets['type'],
                gap_thresholds=gap_thresholds,
                max_duration=self.config['s2_merge_max_duration'],
                max_gap=max_gap,
                max_area=max_area,
            )
            merged_s2s = strax.merge_peaks(
                peaklets,
                start_merge_at, end_merge_at,
                max_buffer=int(self.config['s2_merge_max_duration']
                               // peaklets['dt'].min()))
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
@strax.takes_config(
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"),
    strax.Option('merge_without_s1', default=True,
                 help="If true, S1s will be igored during the merging. "
                      "It's now possible for a S1 to be inside a S2 post merging"),
)
class Peaks(strax.Plugin):
    """
    Merge peaklets and merged S2s such that we obtain our peaks
    (replacing all peaklets that were later re-merged as S2s). As this
    step is computationally trivial, never save this plugin.
    """
    depends_on = ('peaklets', 'peaklet_classification', 'merged_s2s')
    data_kind = 'peaks'
    provides = 'peaks'
    parallel = True
    save_when = strax.SaveWhen.EXPLICIT

    __version__ = '0.1.2'

    def infer_dtype(self):
        return self.deps['peaklets'].dtype_for('peaklets')

    def compute(self, peaklets, merged_s2s):
        # Remove fake merged S2s from dirty hack, see above
        merged_s2s = merged_s2s[merged_s2s['type'] != FAKE_MERGED_S2_TYPE]
        
        if self.config['merge_without_s1']:
            is_s1 = peaklets['type'] == 1
            peaks = strax.replace_merged(peaklets[~is_s1], merged_s2s)
            peaks = strax.sort_by_time(np.concatenate([peaklets[is_s1],
                                                       peaks]))
        else:
            peaks = strax.replace_merged(peaklets, merged_s2s)

        if self.config['diagnose_sorting']:
            assert np.all(np.diff(peaks['time']) >= 0), "Peaks not sorted"
            if self.config['merge_without_s1']:
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
    """Calculates the tight coincidence based on hits and PMT channels.

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

    :returns: n_coin_hit, n_coin_channel of length peaks containing the
        tight coincidence.
    """
    left_hit_i = 0
    n_coin_hit = np.zeros(len(peak_max_times), dtype=np.int16)
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
                n_coin_hit[p_i] += 1
                channels_seen[hit_channel[left_hit_i]-start_ch] = 1

            # stop the loop when we know we're outside the range
            if d > right:
                n_coin_channel[p_i] = np.sum(channels_seen)
                break
        
        # Add channel information in case there are no hits beyond 
        # the last peak:
        n_coin_channel[p_i] = np.sum(channels_seen)

    return n_coin_hit, n_coin_channel


@numba.njit(cache=True, nogil=True)
def hit_max_sample(records, hits):
    """Return the index of the maximum sample for hits"""
    result = np.zeros(len(hits), dtype=np.int16)
    for i, h in enumerate(hits):
        r = records[h['record_i']]
        w = r['data'][h['left']:h['right']]
        result[i] = np.argmax(w)
    return result
