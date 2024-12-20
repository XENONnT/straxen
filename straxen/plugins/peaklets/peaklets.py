from typing import Dict, Tuple, Union
import numba
import numpy as np
from immutabledict import immutabledict
import strax
from strax.processing.general import _touching_windows
from strax.dtypes import DIGITAL_SUM_WAVEFORM_CHANNEL
import straxen


export, __all__ = strax.exporter()


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

    depends_on = "records"
    provides: Union[Tuple[str, ...], str] = ("peaklets", "lone_hits")
    data_kind: Union[Dict[str, str], str] = dict(peaklets="peaklets", lone_hits="lone_hits")
    parallel = "process"
    compressor = "zstd"

    __version__ = "1.2.1"

    peaklet_gap_threshold = straxen.URLConfig(
        default=700, infer_type=False, help="No hits for this many ns triggers a new peak"
    )

    peak_left_extension = straxen.URLConfig(
        default=30, infer_type=False, help="Include this many ns left of hits in peaks"
    )

    peak_right_extension = straxen.URLConfig(
        default=200, infer_type=False, help="Include this many ns right of hits in peaks"
    )

    peak_min_pmts = straxen.URLConfig(
        default=2,
        infer_type=False,
        help="Minimum number of contributing PMTs needed to define a peak",
    )

    peak_split_gof_threshold = straxen.URLConfig(
        # See https://xe1t-wiki.lngs.infn.it/doku.php?id=
        # xenon:xenonnt:analysis:strax_clustering_classification
        # #natural_breaks_splitting
        # for more information
        default=(None, ((0.5, 1.0), (6.0, 0.4)), ((2.5, 1.0), (5.625, 0.4))),  # Reserved
        infer_type=False,
        help=(
            "Natural breaks goodness of fit/split threshold to split "
            "a peak. Specify as tuples of (log10(area), threshold)."
        ),
    )

    peak_split_filter_wing_width = straxen.URLConfig(
        default=70,
        infer_type=False,
        help="Wing width of moving average filter for low-split natural breaks",
    )

    peak_split_min_area = straxen.URLConfig(
        default=40.0,
        infer_type=False,
        help="Minimum area to evaluate natural breaks criterion. Smaller peaks are not split.",
    )

    peak_split_iterations = straxen.URLConfig(
        default=20, infer_type=False, help="Maximum number of recursive peak splits to do."
    )

    diagnose_sorting = straxen.URLConfig(
        track=False,
        default=False,
        infer_type=False,
        help="Enable runtime checks for sorting and disjointness",
    )

    gain_model = straxen.URLConfig(
        infer_type=False, help="PMT gain model. Specify as URL or explicit value"
    )

    tight_coincidence_window_left = straxen.URLConfig(
        default=50,
        infer_type=False,
        help="Time range left of peak center to call a hit a tight coincidence (ns)",
    )

    tight_coincidence_window_right = straxen.URLConfig(
        default=50,
        infer_type=False,
        help="Time range right of peak center to call a hit a tight coincidence (ns)",
    )

    n_tpc_pmts = straxen.URLConfig(type=int, help="Number of TPC PMTs")

    n_top_pmts = straxen.URLConfig(type=int, help="Number of top TPC array PMTs")

    store_data_top = straxen.URLConfig(
        default=True, type=bool, help="Save the sum waveform of the top array separately"
    )

    store_data_start = straxen.URLConfig(
        default=True, type=bool, help="Save the start time of the waveform with 10 ns dt"
    )

    saturation_correction_on = straxen.URLConfig(
        default=True, infer_type=False, help="On off switch for saturation correction"
    )

    saturation_reference_length = straxen.URLConfig(
        default=100,
        infer_type=False,
        help="Maximum number of reference sample used to correct saturated samples",
    )

    saturation_min_reference_length = straxen.URLConfig(
        default=20,
        infer_type=False,
        help="Minimum number of reference sample used to correct saturated samples",
    )

    peaklet_max_duration = straxen.URLConfig(
        default=int(10e6), infer_type=False, help="Maximum duration [ns] of a peaklet"
    )

    channel_map = straxen.URLConfig(
        track=False,
        type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) channel number.",
    )

    hit_min_amplitude = straxen.URLConfig(
        track=True,
        infer_type=False,
        default="cmt://hit_thresholds_tpc?version=ONLINE&run_id=plugin.run_id",
        help=(
            "Minimum hit amplitude in ADC counts above baseline. "
            "Specify as a tuple of length n_tpc_pmts, or a number, "
            "or a tuple like (correction=str, version=str, nT=boolean),"
            "which means we are using cmt."
        ),
    )

    def infer_dtype(self):
        return dict(
            peaklets=strax.peak_dtype(
                n_channels=self.n_tpc_pmts,
                store_data_top=self.store_data_top,
                store_data_start=self.store_data_start,
            ),
            lone_hits=strax.hit_dtype,
        )

    def setup(self):
        if self.peak_min_pmts > 2:
            # Can fix by re-splitting,
            raise NotImplementedError(
                f"Raising the peak_min_pmts to {self.peak_min_pmts} "
                "interferes with lone_hit definition. "
                "See github.com/XENONnT/straxen/issues/295"
            )

        self.to_pe = self.gain_model

        self.hit_thresholds = self.hit_min_amplitude

        self.channel_range = self.channel_map["tpc"]

    def compute(self, records, start, end):
        hits = strax.find_hits(records, min_amplitude=self.hit_thresholds)

        # Remove hits in zero-gain channels
        # they should not affect the clustering!
        hits = hits[self.to_pe[hits["channel"]] != 0]

        hits = strax.sort_by_time(hits)

        # Use peaklet gap threshold for initial clustering
        # based on gaps between hits
        peaklets = strax.find_peaks(
            hits,
            self.to_pe,
            gap_threshold=self.peaklet_gap_threshold,
            left_extension=self.peak_left_extension,
            right_extension=self.peak_right_extension,
            min_channels=self.peak_min_pmts,
            # NB, need to have the data_top field here, will discard if not digitized later
            result_dtype=strax.peak_dtype(
                n_channels=self.n_tpc_pmts, store_data_top=True, store_data_start=True
            ),
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
            lone_hits,
            records,
            peaklets,
            save_outside_hits=(self.peak_left_extension, self.peak_right_extension),
            n_channels=len(self.to_pe),
        )
        if np.any(lone_hits["right_integration"] - lone_hits["left_integration"] <= 0):
            raise ValueError("Find lone_hits with non-positive length!")

        # Compute basic peak properties -- needed before natural breaks
        hits = hits[~is_lone_hit]
        # Define regions outside of peaks such that _find_hit_integration_bounds
        # is not extended beyond a peak.
        outside_peaks = self.create_outside_peaks_region(peaklets, start, end)
        strax.find_hit_integration_bounds(
            hits,
            outside_peaks,
            records,
            save_outside_hits=(self.peak_left_extension, self.peak_right_extension),
            n_channels=len(self.to_pe),
            allow_bounds_beyond_records=True,
        )

        # Transform hits to hitlets for naming conventions. A hit refers
        # to the central part above threshold a hitlet to the entire signal
        # including the left and right extension.
        # (We are not going to use the actual hitlet data_type here.)
        hitlets = hits
        del hits

        # Extend hits into hitlets and clip at chunk boundaries:
        hitlets["time"] -= (hitlets["left"] - hitlets["left_integration"]) * hitlets["dt"]
        hitlets["length"] = hitlets["right_integration"] - hitlets["left_integration"]

        hitlets = strax.sort_by_time(hitlets)
        self.clip_peaklet_times(hitlets, start, end)
        rlinks = strax.record_links(records)

        strax.sum_waveform(
            peaklets,
            hitlets,
            records,
            rlinks,
            self.to_pe,
            n_top_channels=self.n_top_pmts,
            store_data_top=self.store_data_top,
            store_data_start=self.store_data_start,
        )

        strax.compute_properties(peaklets, n_top_channels=self.n_top_pmts)

        # Split peaks using low-split natural breaks;
        # see https://github.com/XENONnT/straxen/pull/45
        # and https://github.com/AxFoundation/strax/pull/225
        peaklets = strax.split_peaks(
            peaklets,
            hitlets,
            records,
            rlinks,
            self.to_pe,
            algorithm="natural_breaks",
            threshold=self.natural_breaks_threshold,
            split_low=True,
            filter_wing_width=self.peak_split_filter_wing_width,
            min_area=self.peak_split_min_area,
            do_iterations=self.peak_split_iterations,
            n_top_channels=self.n_top_pmts,
            store_data_top=self.store_data_top,
            store_data_start=self.store_data_start,
        )

        # Saturation correction using non-saturated channels
        # similar method used in pax
        # see https://github.com/XENON1T/pax/pull/712

        if self.saturation_correction_on:
            peak_list = peak_saturation_correction(
                records,
                rlinks,
                peaklets,
                hitlets,
                self.to_pe,
                reference_length=self.saturation_reference_length,
                min_reference_length=self.saturation_min_reference_length,
                n_top_channels=self.n_top_pmts,
                store_data_top=self.store_data_top,
                store_data_start=self.store_data_start,
            )

            # Compute the width again for corrected peaks
            strax.compute_properties(
                peaklets, n_top_channels=self.n_top_pmts, select_peaks_indices=peak_list
            )

        # Compute tight coincidence level.
        # Making this a separate plugin would
        # (a) doing hitfinding yet again (or storing hits)
        # (b) increase strax memory usage / max_messages,
        #     possibly due to its currently primitive scheduling.
        hit_max_times_argsort = strax.stable_argsort(hitlets["max_time"])
        sorted_hit_max_times = hitlets["max_time"][hit_max_times_argsort]
        sorted_hit_channels = hitlets["channel"][hit_max_times_argsort]
        peaklet_max_times = peaklets["time"] + np.argmax(peaklets["data"], axis=1) * peaklets["dt"]
        peaklets["tight_coincidence"] = get_tight_coin(
            sorted_hit_max_times,
            sorted_hit_channels,
            peaklet_max_times,
            self.tight_coincidence_window_left,
            self.tight_coincidence_window_right,
            self.channel_range,
        )

        # Add max and min time difference between apexes of hits
        self.add_hit_features(hitlets, peaklets)

        if self.diagnose_sorting and len(records):
            assert np.diff(records["time"]).min(initial=1) >= 0, "Records not sorted"
            assert np.diff(hitlets["time"]).min(initial=1) >= 0, "Hits/Hitlets not sorted"
            assert np.all(
                peaklets["time"][1:] >= strax.endtime(peaklets)[:-1]
            ), "Peaks not disjoint"

        # Update nhits of peaklets:
        counts = strax.touching_windows(hitlets, peaklets)
        counts = np.diff(counts, axis=1).flatten()
        peaklets["n_hits"] = counts

        # Drop the data_top or data_start field
        if (not self.store_data_top) or (not self.store_data_start):
            peaklets = drop_data_field(peaklets, self.dtype_for("peaklets"))

        # Check channel of peaklets
        peaklets_unique_channel = np.unique(peaklets["channel"])
        if (peaklets_unique_channel == DIGITAL_SUM_WAVEFORM_CHANNEL).sum() > 1:
            raise ValueError(
                f"Found channel number of peaklets other than {DIGITAL_SUM_WAVEFORM_CHANNEL}"
            )
        # Check tight_coincidence
        if not np.all(peaklets["n_hits"] >= peaklets["tight_coincidence"]):
            raise ValueError(f"Found n_hits less than tight_coincidence")

        return dict(peaklets=peaklets, lone_hits=lone_hits)

    def natural_breaks_threshold(self, peaks):
        rise_time = -peaks["area_decile_from_midpoint"][:, 1]

        # This is ~1 for an clean S2, ~0 for a clean S1,
        # and transitions gradually in between.
        f_s2 = 8 * np.log10(rise_time.clip(1, 1e5) / 100)
        f_s2 = 1 / (1 + np.exp(-f_s2))

        log_area = np.log10(peaks["area"].clip(1, 1e7))
        thresholds = self.peak_split_gof_threshold
        return f_s2 * np.interp(log_area, *np.transpose(thresholds[2])) + (1 - f_s2) * np.interp(
            log_area, *np.transpose(thresholds[1])
        )

    @staticmethod
    @numba.njit(nogil=True, cache=True)
    def clip_peaklet_times(peaklets, start, end):
        for p in peaklets:
            if p["time"] < start:
                p["time"] = start
            if strax.endtime(p) > end:
                p["length"] = (end - p["time"]) // p["dt"]

    @staticmethod
    def create_outside_peaks_region(peaklets, start, end):
        """Creates time intervals which are outside peaks.

        :param peaklets: Peaklets for which intervals should be computed.
        :param start: Chunk start
        :param end: Chunk end
        :return: array of strax.time_fields dtype.

        """
        if not len(peaklets):
            return np.zeros(0, dtype=strax.time_fields)

        outside_peaks = np.zeros(len(peaklets) + 1, dtype=strax.time_fields)

        outside_peaks[0]["time"] = start
        outside_peaks[0]["endtime"] = peaklets[0]["time"]
        outside_peaks[1:-1]["time"] = strax.endtime(peaklets[:-1])
        outside_peaks[1:-1]["endtime"] = peaklets["time"][1:]
        outside_peaks[-1]["time"] = strax.endtime(peaklets[-1])
        outside_peaks[-1]["endtime"] = end
        return outside_peaks

    @staticmethod
    def add_hit_features(hitlets, peaklets):
        """Create hits timing features."""
        split_hits = strax.split_by_containment(hitlets, peaklets)
        for peaklet, h_max in zip(peaklets, split_hits):
            max_time_diff = np.diff(strax.stable_sort(h_max["max_time"]))
            if len(max_time_diff) > 0:
                peaklet["max_diff"] = max_time_diff.max()
                peaklet["min_diff"] = max_time_diff.min()
            else:
                peaklet["max_diff"] = -1
                peaklet["min_diff"] = -1


def drop_data_field(peaklets, goal_dtype, _name_function="_drop_data_field"):
    """Return peaklets without the data_* field."""
    peaklets_without_field = np.zeros(len(peaklets), dtype=goal_dtype)
    strax.copy_to_buffer(peaklets, peaklets_without_field, _name_function)
    del peaklets
    return peaklets_without_field


@numba.jit(nopython=True, nogil=True, cache=False)
def peak_saturation_correction(
    records,
    rlinks,
    peaks,
    hitlets,
    to_pe,
    reference_length=100,
    min_reference_length=20,
    use_classification=False,
    n_top_channels=0,
    store_data_top=False,
    store_data_start=False,
):
    """Correct the area and per pmt area of peaks from saturation.

    :param records: Records
    :param rlinks: strax.record_links of corresponding records.
    :param peaks: Peaklets / Peaks
    :param hitlets: Hitlets found in records to build peaks. (Hitlets are hits including the
        left/right extension)
    :param to_pe: adc to PE conversion (length should equal number of PMTs)
    :param reference_length: Maximum number of reference sample used to correct saturated samples
    :param min_reference_length: Minimum number of reference sample used to correct saturated
        samples
    :param use_classification: Option of using classification to pick only S2
    :param n_top_channels: Number of top array channels.
    :param store_data_top: Boolean which indicates whether to store the top array waveform in the
        peak.
    :param store_data_start: Boolean which indicates whether to store the first samples of the
        waveform in the peak.

    """

    if not len(records):
        return
    if not len(peaks):
        return

    # Search for peaks with saturated channels
    mask = peaks["n_saturated_channels"] > 0
    if use_classification:
        mask &= peaks["type"] == 2
    peak_list = np.where(mask)[0]
    # Look up records that touch each peak
    record_ranges = _touching_windows(
        records["time"],
        strax.endtime(records),
        peaks[peak_list]["time"],
        strax.endtime(peaks[peak_list]),
    )

    # Create temporary arrays for calculation
    dt = records[0]["dt"]
    n_channels = len(peaks[0]["saturated_channel"])
    len_buffer = np.max(peaks["length"] * peaks["dt"]) // dt + 1
    max_nrecord = len_buffer // len(records[0]["data"]) + 1

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
        channel_saturated = p["saturated_channel"] > 0

        for record_i in range(record_ranges[ix][0], record_ranges[ix][1]):
            r = records[record_i]
            r_slice, b_slice = strax.overlap_indices(
                r["time"] // dt, r["length"], p["time"] // dt, p["length"] * p["dt"] // dt
            )

            ch = r["channel"]
            if channel_saturated[ch]:
                b_pulse[ch, slice(*b_slice)] += r["data"][slice(*r_slice)]
                b_index[ch, np.argmin(b_index[ch])] = record_i
            else:
                b_sumwf[slice(*b_slice)] += r["data"][slice(*r_slice)] * to_pe[ch]

        _peak_saturation_correction_inner(
            channel_saturated,
            records,
            p,
            to_pe,
            b_sumwf,
            b_pulse,
            b_index,
            reference_length,
            min_reference_length,
        )

        # Back track sum wf downsampling
        peaks[peak_i]["length"] = p["length"] * p["dt"] / dt
        peaks[peak_i]["dt"] = dt

    strax.sum_waveform(
        peaks,
        hitlets,
        records,
        rlinks,
        to_pe,
        n_top_channels=n_top_channels,
        store_data_top=store_data_top,
        store_data_start=store_data_start,
        select_peaks_indices=peak_list,
    )
    return peak_list


@numba.jit(nopython=True, nogil=True, cache=True)
def _peak_saturation_correction_inner(
    channel_saturated,
    records,
    p,
    to_pe,
    b_sumwf,
    b_pulse,
    b_index,
    reference_length=100,
    min_reference_length=20,
):
    """Would add a third level loop in peak_saturation_correction Which is not ideal for numba, thus
    this function is written.

    :param channel_saturated: (bool, n_channels)
    :param p: One peak/peaklet
    :param to_pe: adc to PE conversion (length should equal number of PMTs)
    :param b_sumwf b_pulse b_index: Filled buffers

    """
    dt = records["dt"][0]
    n_channels = len(channel_saturated)

    for ch in range(n_channels):
        if not channel_saturated[ch]:
            continue
        b = b_pulse[ch]

        # Define the reference region as reference_length before the first saturation point
        # unless there are not enough samples
        bl = np.inf
        for record_i in b_index[ch]:
            if record_i == -1:
                break
            bl = min(bl, records["baseline"][record_i])

        s0 = np.argmax(b >= np.int16(bl))
        ref = slice(max(0, s0 - reference_length), s0)

        if (b[ref] * to_pe[ch] > 1).sum() < min_reference_length:
            # the pulse is saturated, but there are not enough reference samples to get a good ratio
            # This actually distinguished between S1 and S2 and will only correct S2 signals
            continue
        if (b_sumwf[ref] > 1).sum() < min_reference_length:
            # the same condition applies to the waveform model
            continue
        if np.sum(b[ref]) * to_pe[ch] / np.sum(b_sumwf[ref]) > 1:
            # The pulse is saturated,
            # but insufficient information is available in the other channels
            # to reliably reconstruct it
            continue

        scale = np.sum(b[ref]) / np.sum(b_sumwf[ref])

        # Loop over the record indices of the saturated channel (saved in b_index buffer)
        for record_i in b_index[ch]:
            if record_i == -1:
                break
            r = records[record_i]
            r_slice, b_slice = strax.overlap_indices(
                r["time"] // dt, r["length"], p["time"] // dt + s0, p["length"] * p["dt"] // dt - s0
            )

            if r_slice[1] == r_slice[0]:  # This record proceeds saturation
                continue
            b_slice = b_slice[0] + s0, b_slice[1] + s0

            # First is finding the highest point in the desaturated record
            # because we need to bit shift the whole record if it exceeds int16 range
            apax = scale * max(b_sumwf[slice(*b_slice)])

            if np.int32(apax) >= 2**15:  # int16(2**15) is -2**15
                bshift = int(np.floor(np.log2(apax) - 14))

                tmp = r["data"].astype(np.int32)
                tmp[slice(*r_slice)] = b_sumwf[slice(*b_slice)] * scale

                r["area"] = np.sum(tmp)  # Auto covert to int64
                r["data"][:] = np.right_shift(tmp, bshift)
                r["amplitude_bit_shift"] += bshift
            else:
                r["data"][slice(*r_slice)] = b_sumwf[slice(*b_slice)] * scale
                r["area"] = np.sum(r["data"])


@numba.jit(nopython=True, nogil=True, cache=True)
def get_tight_coin(hit_max_times, hit_channel, peak_max_times, left, right, channels=(0, 493)):
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

    :return: n_coin_channel of length peaks containing the
        tight coincidence.

    """
    left_hit_i = 0
    n_coin_channel = np.zeros(len(peak_max_times), dtype=np.int16)
    start_ch, end_ch = channels
    channels_seen = np.zeros(end_ch - start_ch + 1, dtype=np.bool_)

    # loop over peaks
    for p_i, p_t in enumerate(peak_max_times):
        channels_seen[:] = 0
        # loop over hits starting from the last one we left at
        for left_hit_i in range(left_hit_i, len(hit_max_times)):
            # if the hit is in the window, its a tight coin
            d = hit_max_times[left_hit_i] - p_t
            if (-left <= d) & (d <= right):
                channels_seen[hit_channel[left_hit_i] - start_ch] = 1

            # stop the loop when we know we're outside the range
            if d > right:
                n_coin_channel[p_i] = np.sum(channels_seen)
                break

        # Add channel information in case there are no hits beyond
        # the last peak:
        n_coin_channel[p_i] = np.sum(channels_seen)

    return n_coin_channel
