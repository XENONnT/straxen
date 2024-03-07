import numba
import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@export
class LEDAfterpulseProcessing(strax.Plugin):
    """
    Plugin for processing LED afterpulses.

    Detect LED pulses and afterpulses (APs) in raw_records waveforms. Compute 
    the AP datatype.

    """

    __version__ = "0.6.0"
    depends_on = "raw_records"
    data_kind = "afterpulses"
    provides = "afterpulses"
    compressor = "zstd"
    parallel = "process"
    rechunk_on_save = True

    gain_model = straxen.URLConfig(
        infer_type=False,
        help="PMT gain model. Specify as (model_type, model_config)",
    )

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help="Number of PMTs in TPC",
    )

    LED_hit_left_boundary = straxen.URLConfig(
        default=50,
        infer_type=False,
        help="Left boundary after which the first hit marks the position of the LED window",
    )

    LED_hit_right_boundary = straxen.URLConfig(
        default=110,
        infer_type=False,
        help="Right boundary after which the LED hit will no longer be searched",
    )

    LED_window_width = straxen.URLConfig(
        default=20,
        infer_type=False,
        help="Width of the window in which hits are merged into the LED hit after the first hit",
    )

    baseline_samples = straxen.URLConfig(
        default=40,
        infer_type=False,
        help="Number of samples to use at start of WF to determine the baseline",
    )

    hit_min_amplitude = straxen.URLConfig(
        track=True,
        infer_type=False,
        default="cmt://hit_thresholds_tpc?version=ONLINE&run_id=plugin.run_id",
        help=(
            "Minimum hit amplitude in ADC counts above baseline. "
            "Specify as a tuple of length n_tpc_pmts, or a number,"
            'or a string like "legacy-thresholds://pmt_commissioning_initial" which means calling'
            "hitfinder_thresholds.py"
            'or url string like "cmt://hit_thresholds_tpc?version=ONLINE" which means'
            "calling cmt."
        ),
    )

    hit_min_height_over_noise = straxen.URLConfig(
        default=4,
        infer_type=False,
        help=(
            "Minimum hit amplitude in numbers of baseline_rms above baseline."
            "Actual threshold used is max(hit_min_amplitude, hit_min_"
            "height_over_noise * baseline_rms)."
        ),
    )

    save_outside_hits = straxen.URLConfig(
        default=(3, 20),
        infer_type=False,
        help="Save (left, right) samples besides hits; cut the rest",
    )

    def infer_dtype(self):
        dtype = dtype_afterpulses()
        return dtype

    def setup(self):
        self.to_pe = self.gain_model
        self.hit_thresholds = self.hit_min_amplitude
        self.hit_left_extension, self.hit_right_extension = self.save_outside_hits

    def compute(self, raw_records):
        # Convert everything to the records data type -- adds extra fields.
        records = strax.raw_to_records(raw_records)
        del raw_records

        # calculate baseline and baseline rms
        strax.baseline(records, baseline_samples=self.baseline_samples, flip=True)

        # find all hits
        hits = strax.find_hits(
            records,
            min_amplitude=self.hit_thresholds,
            min_height_over_noise=self.hit_min_height_over_noise,
        )

        # sort hits by record_i and time, then find LED hit and afterpulse
        # hits within the same record
        hits_ap = find_ap(
            hits,
            records,
            LED_hit_left_boundary=self.LED_hit_left_boundary,
            LED_hit_right_boundary=self.LED_hit_right_boundary,
            LED_window_width=self.LED_window_width,
            hit_left_extension=self.hit_left_extension,
            hit_right_extension=self.hit_right_extension,
        )

        hits_ap["area_pe"] = hits_ap["area"] * self.to_pe[hits_ap["channel"]]
        hits_ap["height_pe"] = hits_ap["height"] * self.to_pe[hits_ap["channel"]]

        return hits_ap


@export
def find_ap(
    hits, 
    records, 
    LED_hit_left_boundary,
    LED_hit_right_boundary,
    LED_window_width, 
    hit_left_extension, 
    hit_right_extension
):
    """
    Find afterpulses (APs) in the given hits data within specified LED hit 
    boundaries and extensions.

    Parameters
    ----------
    hits :
        Array containing hit data.
    records :
        Array containing record data.
    LED_hit_left_boundary :
        Left boundary of the LED hit window. 
    LED_hit_right_boundary :
        Right boundary of the LED hit window.
    LED_window_width :
        Extension to the right of the first hit found in the LED hit window 
        within which hits are merged into the LED hit.
    hit_left_extension :
        Extension to the left of the hit window.
    hit_right_extension :
        Extension to the right of the hit window.

    Returns
    -------
    Array containing afterpulse data. 

    Notes
    -----
    - Hits to the left of the LED_hit_left_boundary are ignored.
    - If no hit is found between LED_hit_left_boundary and LED_hit_right_boundary
    the record is skipped.
    - The merged LED hits are also saved and can be selected for by having 
    t_delay = 0 by definition.
    """
    
    buffer = np.zeros(len(hits), dtype=dtype_afterpulses())

    if not len(hits):
        return buffer

    # sort hits first by record_i, then by time
    hits_sorted = np.sort(hits, order=("record_i", "time"))
    res = _find_ap(
        hits_sorted,
        records,
        LED_hit_left_boundary,
        LED_hit_right_boundary,
        LED_window_width,
        hit_left_extension,
        hit_right_extension,
        buffer=buffer,
    )
    return res


@numba.jit(nopython=True, nogil=True, cache=True)
def _find_ap(
    hits,
    records,
    LED_hit_left_boundary,
    LED_hit_right_boundary,
    LED_window_width,
    hit_left_extension,
    hit_right_extension,
    buffer=None,
):
    # hits need to be sorted by record_i, then time!
    offset = 0

    is_LED = False
    t_LED = None
    t_LED_hit = None

    prev_record_i = hits[0]["record_i"]
    record_data = records[prev_record_i]["data"]
    record_len = records[prev_record_i]["length"]
    baseline_fpart = records[prev_record_i]["baseline"] % 1

    for h_i, h in enumerate(hits):
        if h["record_i"] > prev_record_i:
            # start of a new record
            is_LED = False
            # only increment buffer if the old one is not empty! this happens
            # when no (LED) hit is found in the previous record
            if not buffer[offset]["time"] == 0:
                offset += 1
            prev_record_i = h["record_i"]
            record_data = records[prev_record_i]["data"]
            baseline_fpart = records[prev_record_i]["baseline"] % 1

        res = buffer[offset]

        if h["left"] < LED_hit_left_boundary:
            # if hit is before LED hit window: discard
            continue
        
        if (h["left"] < LED_hit_right_boundary) and not is_LED:
            # this is the first hit in the LED hit window
            fill_hitpars(
                res,
                h,
                hit_left_extension,
                hit_right_extension,
                record_data,
                record_len,
                baseline_fpart,
            )

            t_LED_hit = res["sample_10pc_area"]
            t_LED = res["sample_10pc_area"]
            is_LED = True

            continue
        
        if not is_LED:
            # No hit found in the LED hit window: skip to next record
            continue

        if h["left"] < (t_LED_hit + LED_window_width):
            # This hit is still inside the LED window: extend the LED hit
            fill_hitpars(
                res,
                h,
                hit_left_extension,
                hit_right_extension,
                record_data,
                record_len,
                baseline_fpart,
                extend=True
            )

            # set the LED time in the current WF
            t_LED = res["sample_10pc_area"]

            continue

        # Here begins a new hit after the LED window

        # if a hit is completely inside the previous hit's right_extension,
        # then skip it (because it's already included in the previous hit)
        if h["right"] <= res["right_integration"]:
            continue

        # if a hit only partly overlaps with the previous hit's right_
        # extension, merge them (extend previous hit by this one)
        if h["left"] <= res["right_integration"]:
            fill_hitpars(
                res,
                h,
                hit_left_extension,
                hit_right_extension,
                record_data,
                record_len,
                baseline_fpart,
                extend=True,
            )

            res["tdelay"] = res["sample_10pc_area"] - t_LED

            continue

        # an actual new hit increases the buffer index
        offset += 1
        res = buffer[offset]

        fill_hitpars(
            res, 
            h, 
            hit_left_extension, 
            hit_right_extension, 
            record_data, 
            record_len, 
            baseline_fpart
        )

        res["tdelay"] = res["sample_10pc_area"] - t_LED

    return buffer[:offset]


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def get_sample_area_quantile(data, quantile, baseline_fpart):

    """
    Return the index of the first sample in the hit where the integrated area 
    of the hit to that index is above the specified quantile of the total area.

    Parameters:
    - data : 
        Array containing the baselined waveform data of the hit.
    - quantile : 
        The quantile (0 to 1) representing the threshold for the area.
    - baseline_fpart : 
        Fractional part of the baseline (baseline % 1) of the record

    Return:
    - int: The index of the first sample where the area exceeds the quantile of the total area.
           If no such sample is found, returns 0.

    Notes:
    - If no quantile is found where the area exceeds the threshold, it returns 0. This is
    usually caused by real events in the baseline window, which can result in a negative
    area.
    """

    area = 0
    area_tot = data.sum() + len(data) * baseline_fpart

    for d_i, d in enumerate(data):
        area += d + baseline_fpart
        if area > (quantile * area_tot):
            return d_i
        if d_i == len(data) - 1:
            # if no quantile was found, something went wrong
            # (negative area due to wrong baseline, caused by real events that
            # by coincidence fall in the first samples of the trigger window)
            # print('no quantile found: set to 0')
            return 0

    # What happened here?!
    return 0


@numba.jit(nopython=True, nogil=True, cache=True)
def fill_hitpars(
    result,
    hit,
    hit_left_extension,
    hit_right_extension,
    record_data,
    record_len,
    baseline_fpart,
    extend=False,
):
    if not extend:  # fill first time only
        result["time"] = hit["time"] - hit_left_extension * hit["dt"]
        result["dt"] = hit["dt"]
        result["channel"] = hit["channel"]
        result["left"] = hit["left"]
        result["record_i"] = hit["record_i"]
        result["threshold"] = hit["threshold"]
        result["left_integration"] = hit["left"] - hit_left_extension
        result["height"] = hit["height"]

    # fill always (if hits are merged, only these will be updated)
    result["right"] = hit["right"]
    result["right_integration"] = hit["right"] + hit_right_extension
    if result["right_integration"] > record_len:
        result["right_integration"] = record_len  # cap right_integration at end of record
    result["length"] = result["right_integration"] - result["left_integration"]

    hit_data = record_data[result["left_integration"] : result["right_integration"]]
    result["area"] = hit_data.sum() + result["length"] * baseline_fpart
    result["sample_10pc_area"] = result["left_integration"] + get_sample_area_quantile(
        hit_data, 0.1, baseline_fpart
    )
    result["sample_50pc_area"] = result["left_integration"] + get_sample_area_quantile(
        hit_data, 0.5, baseline_fpart
    )
    if len(hit_data):
        result["max"] = result["left_integration"] + hit_data.argmax()

    if extend:  # only when merging hits
        result["height"] = max(result["height"], hit["height"])


@export
def dtype_afterpulses():
    """The afterpulse datatype

    Return:
    - The afterpulse datatype
    """
    dtype_ap = [
        (("Channel/PMT number", "channel"), "<i2"),
        (("Time resolution in ns", "dt"), "<i2"),
        (("Start time of the interval (ns since unix epoch)", "time"), "<i8"),
        (("Length of the interval in samples", "length"), "<i4"),
        (("Integral in ADC x samples", "area"), "<i4"),
        (("Pulse area in PE", "area_pe"), "<f4"),
        (("Sample index in which hit starts", "left"), "<i2"),
        (("Sample index in which hit area succeeds 10% of total area", "sample_10pc_area"), "<i2"),
        (("Sample index in which hit area succeeds 50% of total area", "sample_50pc_area"), "<i2"),
        (("Sample index of hit maximum", "max"), "<i2"),
        (("Index of first sample in record just beyond hit (exclusive bound)", "right"), "<i2"),
        (("Height of hit in ADC counts", "height"), "<i4"),
        (("Height of hit in PE", "height_pe"), "<f4"),
        (("Delay of hit w.r.t. LED hit in same WF, in samples", "tdelay"), "<i2"),
        (("Internal (temporary) index of fragment in which hit was found", "record_i"), "<i4"),
        (("Index of sample in record where integration starts", "left_integration"), np.int16),
        (("Index of first sample beyond integration region", "right_integration"), np.int16),
        (("ADC threshold applied in order to find hits", "threshold"), np.float32),
    ]
    return dtype_ap
