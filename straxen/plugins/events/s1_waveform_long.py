import numba
import numpy as np
import strax
import straxen
from straxen.plugins.peaklets.peaklets import Peaklets

export, __all__ = strax.exporter()

N_WAVEFORM_SAMPLES = 512
WAVEFORM_LEFT_WINDOW = 256
WAVEFORM_DT = 10

S1_WINDOW_SAVETY_FACTOR = 10

tmp_container_dtype = [
    ("time", np.int64),
    ("dt", np.int64),
    ("length", np.int64),
    ("s1_data_long", (np.float32, N_WAVEFORM_SAMPLES)),
    ("s1_data_long_top", (np.float32, N_WAVEFORM_SAMPLES)),
    ("s1_data_long_type_mask", (np.int16, N_WAVEFORM_SAMPLES)),
]

@export
class S1WaveformLong(Peaklets):

    """Build the waveform around the main S1 peak"""

    depends_on = ("records", "peak_basics", "event_basics")

    provides = "s1_waveform_long"

    data_kind = "events"

    child_plugin = True

    __version__ = "0.0.5"


    def infer_dtype(self):

        dtype = [
            ("time", np.int64),
            ("endtime", np.int64),
            ("s1_data_long_dt", np.int64),
            ("s1_data_long_length", np.int64),
            ("s1_data_long_start_time", np.int64),
            ("s1_data_long", (np.float32, N_WAVEFORM_SAMPLES)),
            ("s1_data_long_top", (np.float32, N_WAVEFORM_SAMPLES)),
            ("s1_data_long_type_mask",(np.int16, N_WAVEFORM_SAMPLES)),
        ]

        return dtype

    def compute(self, records, peaks, events, start, end):

        # Remove all records that are far away from the main S1 of a event
        s1_window = np.zeros(
            len(events),
            dtype=strax.interval_dtype
        )

        s1_window["dt"] = WAVEFORM_DT
        s1_window["length"] = N_WAVEFORM_SAMPLES * S1_WINDOW_SAVETY_FACTOR
        s1_window['time'] = events['s1_time'] - WAVEFORM_LEFT_WINDOW * WAVEFORM_DT * S1_WINDOW_SAVETY_FACTOR
        s1_window = strax.sort_by_time(s1_window)

        outside_main_s1 = strax.fully_contained_in(records, s1_window) == -1
        records = records[~outside_main_s1]

        r = records

        hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)

        # Remove hits in zero-gain channels
        # they should not affect the clustering!
        hits = hits[self.to_pe[hits["channel"]] != 0]

        hits = strax.sort_by_time(hits)

        is_lone_hit = strax.fully_contained_in(hits, peaks) == -1

        hits = hits[~is_lone_hit]

        outside_peaks = self.create_outside_peaks_region(peaks, start, end)

        # remove the outside regions that have zero length. This should not happen when doing this with peaklets but here it can show up
        outside_peak_mask = outside_peaks["time"] == outside_peaks["endtime"]
        outside_peaks = outside_peaks[~outside_peak_mask]

        strax.find_hit_integration_bounds(
            hits,
            outside_peaks,
            records,
            save_outside_hits=(self.peak_left_extension, self.peak_right_extension),
            n_channels=len(self.to_pe),
            allow_bounds_beyond_records=True,
        )

        hitlets = hits
        del hits

        # Extend hits into hitlets and clip at chunk boundaries:
        hitlets["time"] -= (hitlets["left"] - hitlets["left_integration"]) * hitlets["dt"]
        hitlets["length"] = hitlets["right_integration"] - hitlets["left_integration"]

        hitlets = strax.sort_by_time(hitlets)
        hitlets_time = np.copy(hitlets["time"])
        self.clip_peaklet_times(hitlets, start, end)
        rlinks = strax.record_links(records)

        tmp_container = np.zeros(
            len(events),
            dtype=tmp_container_dtype
        )

        tmp_container["dt"] = WAVEFORM_DT
        tmp_container["length"] = N_WAVEFORM_SAMPLES
        tmp_container['time'] = events['s1_time'] - WAVEFORM_LEFT_WINDOW * WAVEFORM_DT

        sum_waveform(
            tmp_container, hitlets, r, rlinks, self.to_pe, n_top_channels=self.n_top_pmts
        )

        #Sort containers so we can use touching_windows
        sort_ix = np.argsort(tmp_container["time"])
        tmp_container = tmp_container[sort_ix]

        peak_ix = strax.touching_windows(peaks, tmp_container)

        # Build the peak type mask
        peak_type_mask(tmp_container, peaks, peak_ix)

        #Undo the sorting
        unsort_ix = np.argsort(sort_ix)
        tmp_container = tmp_container[unsort_ix]

        # Finally move the data to the final container
        result_container = np.zeros(
            len(events),
            dtype=self.dtype
        )

        result_container["s1_data_long_dt"] = WAVEFORM_DT
        result_container["s1_data_long_length"] = N_WAVEFORM_SAMPLES
        result_container["time"] = events['time']
        result_container['endtime'] = events['endtime']
        result_container['s1_data_long_start_time'] = events['s1_time'] - WAVEFORM_LEFT_WINDOW * WAVEFORM_DT
        result_container["s1_data_long"] = tmp_container["s1_data_long"]
        result_container["s1_data_long_top"] = tmp_container["s1_data_long_top"]
        result_container["s1_data_long_type_mask"] = tmp_container["s1_data_long_type_mask"]


        return result_container

@numba.njit(nogil=True, cache=True)
def sum_waveform(
    containers, hits, records, record_links, adc_to_pe, n_top_channels=0
):
    """Compute sum waveforms for all containers. Adapted from https://github.com/AxFoundation/strax/blob/848e65ebea181c97e8e40d7b575a6039a96da794/strax/processing/peak_building.py#L247-L399
    """
    if not len(records):
        return
    if not len(containers):
        return
    
    select_containers_indices = np.arange(len(containers))
    
    dt = records[0]["dt"]
    n_samples_record = len(records[0]["data"])
    prev_record_i, next_record_i = record_links

    # Big buffer to hold even largest sum waveforms
    # Need a little more even for downsampling..
    swv_buffer = np.zeros(containers["length"].max() * 2, dtype=np.float32)
    twv_buffer = np.zeros(containers["length"].max() * 2, dtype=np.float32)

    # Hit index for hits in containers
    left_h_i = 0
    # Create hit waveform buffer
    hit_waveform = np.zeros(hits["length"].max(), dtype=np.float32)

    for container_i in select_containers_indices:
        c = containers[container_i]
        # Clear the relevant part of the swv buffer for use
        # (we clear a bit extra for use in downsampling)
        c_length = c["length"]
        swv_buffer[: min(2 * c_length, len(swv_buffer))] = 0
        twv_buffer[: min(2 * c_length, len(twv_buffer))] = 0


        # Find first hit that contributes to this peak
        for left_h_i in range(left_h_i, len(hits)):
            h = hits[left_h_i]
            # TODO: need test that fails if we replace < with <= here
            if c["time"] < h["time"] + h["length"] * dt:
                break
        else:
            # Hits exhausted before peaks exhausted
            # TODO: this is a strange case, maybe raise warning/error?
            break

        # Scan over hits that overlap with peak
        for right_h_i in range(left_h_i, len(hits)):
            h = hits[right_h_i]
            record_i = h["record_i"]
            ch = h["channel"]
            assert c["dt"] == h["dt"], "Hits and peaks must have same dt"

            shift = (c["time"] - h["time"]) // dt
            n_samples_hit = h["length"]
            n_samples_peak = c_length

            if shift <= -n_samples_peak:
                # Hit is completely to the right of the peak;
                # we've seen all overlapping records
                break

            if n_samples_hit <= shift:
                # The (real) data in this record does not actually overlap
                # with the peak
                # (although a previous, longer hit did overlap)
                continue

            # Get overlapping samples between hit and peak:
            (h_start, h_end), (c_start, c_end) = strax.overlap_indices(
                h["time"] // dt, n_samples_hit, c["time"] // dt, n_samples_peak
            )

            hit_waveform[:] = 0

            # Get record which belongs to main part of hit (wo integration bounds):
            r = records[record_i]

            is_saturated = _build_hit_waveform(h, r, hit_waveform)

            # Now check if we also have to go to prev/next record due to integration bounds.
            # If bounds are outside of peak we chop when building the summed waveform later.
            if h["left_integration"] < 0 and prev_record_i[record_i] != -1:
                r = records[prev_record_i[record_i]]
                is_saturated |= _build_hit_waveform(h, r, hit_waveform)

            if h["right_integration"] > n_samples_record and next_record_i[record_i] != -1:
                r = records[next_record_i[record_i]]
                is_saturated |= _build_hit_waveform(h, r, hit_waveform)

            hit_data = hit_waveform[h_start:h_end]
            hit_data *= adc_to_pe[ch]
            
            swv_buffer[c_start:c_end] += hit_data

            if ch < n_top_channels:
                twv_buffer[c_start:c_end] += hit_data

        store_waveform(c, swv_buffer, twv_buffer)



@numba.njit(nogil=True, cache=True)
def store_waveform(
    c, wv_buffer, wv_buffer_top
):  

    c["s1_data_long"][: c["length"]] = wv_buffer[: c["length"]]
    c["s1_data_long_top"][: c["length"]] = wv_buffer_top[: c["length"]]


@numba.njit(cache=True, nogil=True)
def _build_hit_waveform(hit, record, hit_waveform):
    """Adds information for overlapping record and hit to hit_waveform. Updates hit_waveform
    inplace. Result is still in ADC counts.

    :return: Boolean if record saturated within the hit.

    """
    (h_start_record, h_end_record), (r_start, r_end) = strax.overlap_indices(
        hit["time"] // hit["dt"], hit["length"], record["time"] // record["dt"], record["length"]
    )

    # Get record properties:
    record_data = record["data"][r_start:r_end]
    multiplier = 2 ** record["amplitude_bit_shift"]
    bl_fpart = record["baseline"] % 1

    try:
        max_in_record = record_data.max() * multiplier
    except:
        # This can happen if the record is empty
        max_in_record = 0
        print('Error: Empty record, setting max_in_record to 0')

    # Build hit waveform:
    hit_waveform[h_start_record:h_end_record] = multiplier * record_data + bl_fpart

    return np.int8(max_in_record >= np.int16(record["baseline"]))



@numba.njit(nogil=True, cache=True)
def peak_type_mask(containers, peaks, peak_ix):

    if not len(peaks):
        return
    if not len(containers):
        return

    for i in range(len(containers)):

        # peak_slice = slice(*peak_ix[i])
        # peaks_of_container = peaks[peak_slice]
        peaks_of_container = peaks[peak_ix[i][0]:peak_ix[i][1]]

        if not len(peaks_of_container):
            continue

        for pk in peaks_of_container:

            (container_wf_start, container_wf_end), (peak_start, peak_end) = strax.overlap_indices(
                containers[i]["time"] // containers[i]["dt"], containers[i]["length"], pk["time"] // 10, pk["length"] * (pk["dt"] // 10)
            )

            type_to_write = pk["type"]
            # Zero means no peak here. Move the type 0 classification to 3
            if type_to_write == 0:
                type_to_write = 3

            containers[i]["s1_data_long_type_mask"][container_wf_start:container_wf_end] = type_to_write