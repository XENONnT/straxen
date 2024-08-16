"""
Dear nT analyser,
if you want to complain please contact:
    chiara@physik.uzh.ch,
    gvolta@physik.uzh.ch,
    kazama@isee.nagoya-u.ac.jp
    torben.flehmke@fysik.su.se
"""

from immutabledict import immutabledict
import strax
import straxen
import numba
import numpy as np

# This makes sure shorthands for only the necessary functions
# are made available under straxen.[...]
export, __all__ = strax.exporter()

channel_list = [i for i in range(494)]


@export
class LEDCalibration(strax.Plugin):
    """
    Preliminary version, several parameters to set during commissioning.
    LEDCalibration returns: channel, time, dt, length, Area,
    amplitudeLED and amplitudeNOISE.

    The new variables are:
        - Area: Area computed in the given window, averaged over 6
          windows that have the same starting sample and different end
          samples.
        - amplitudeLED: peak amplitude of the LED on run in the given
          window.
        - amplitudeNOISE: amplitude of the LED on run in a window far
          from the signal one.
    """

    __version__ = "0.3.0"

    depends_on = "raw_records"
    data_kind = "led_cal"
    compressor = "zstd"
    parallel = "process"
    rechunk_on_save = False

    led_cal_record_length = straxen.URLConfig(
        default=160, infer_type=False, help="Length (samples) of one record without 0 padding."
    )

    baseline_window = straxen.URLConfig(
        default=(0, 40),
        infer_type=False,
        help="Window (samples) for baseline calculation.",
    )

    default_led_window = straxen.URLConfig(
        default=(78, 107),
        infer_type=False,
        help="Default window (samples) to integrate and get the maximum amplitude \
            if no hit was found in the record.",
    )

    led_hit_extension = straxen.URLConfig(
        default=(-5, 24),
        infer_type=False,
        help="The extension around the LED hit to integrate.",
    )

    area_averaging_length = straxen.URLConfig(
        default=10,
        infer_type=False,
        help=(
            "The total length of the averaging window for the area calculation."
            "To mitigate a possiple bias from noise, the area is integrated multiple times with"
            "sligntly different window lengths and then averaged. integration_averaging_length"
            "should be divisible by step."
        ),
    )

    area_averaging_step = straxen.URLConfig(
        default=2,
        infer_type=False,
        help=(
            "The step size used for the different windows, averaged for the area calculation."
            "To mitigate a possiple bias from noise, the area is integrated multiple times with"
            "sligntly different window lengths and then averaged. integration_averaging_length"
            "should be divisible by step."
        ),
    )

    noise_window = straxen.URLConfig(
        default=(10, 48), infer_type=False, help="Window (samples) to analyse the noise"
    )

    channel_list = straxen.URLConfig(
        default=(tuple(channel_list)),
        infer_type=False,
        help="List of PMTs. Defalt value: all the PMTs",
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

    dtype = [
        ("area", np.float32, "Area averaged in integration windows"),
        ("amplitude_led", np.float32, "Amplitude in LED window"),
        ("amplitude_noise", np.float32, "Amplitude in off LED window"),
        ("channel", np.int16, "Channel"),
        ("time", np.int64, "Start time of the interval (ns since unix epoch)"),
        ("dt", np.int16, "Time resolution in ns"),
        ("length", np.int32, "Length of the interval in samples"),
    ]

    def compute(self, raw_records):
        """The data for LED calibration are build for those PMT which belongs to channel list.

        This is used for the different ligh levels. As default value all the PMTs are considered.

        """
        mask = np.where(np.in1d(raw_records["channel"], self.channel_list))[0]
        raw_records_active_channels = raw_records[mask]
        records = get_records(
            raw_records_active_channels, self.baseline_window, self.led_cal_record_length
        )
        del raw_records_active_channels, raw_records

        temp = np.zeros(len(records), dtype=self.dtype)
        strax.copy_to_buffer(records, temp, "_recs_to_temp_led")

        led_windows = get_led_windows(
            records,
            self.default_led_window,
            self.led_hit_extension,
            self.hit_min_height_over_noise,
            self.led_cal_record_length,
            self.area_averaging_length,
        )

        on, off = get_amplitude(records, led_windows, self.noise_window)
        temp["amplitude_led"] = on["amplitude"]
        temp["amplitude_noise"] = off["amplitude"]

        area = get_area(records, led_windows, self.area_averaging_length, self.area_averaging_step)
        temp["area"] = area["area"]
        return temp


def get_records(raw_records, baseline_window, led_cal_record_length):
    """Determine baseline as the average of the first baseline_samples of each pulse.

    Subtract the pulse float(data) from baseline.

    """

    record_length_padded = np.shape(raw_records.dtype["data"])[0]

    _dtype = [
        (("Start time since unix epoch [ns]", "time"), "<i8"),
        (("Length of the interval in samples", "length"), "<i4"),
        (("Width of one sample [ns]", "dt"), "<i2"),
        (("Channel/PMT number", "channel"), "<i2"),
        (
            (
                "Length of pulse to which the record belongs (without zero-padding)",
                "pulse_length",
            ),
            "<i4",
        ),
        (("Fragment number in the pulse", "record_i"), "<i2"),
        (
            ("Baseline in ADC counts. data = int(baseline) - data_orig", "baseline"),
            "f4",
        ),
        (
            ("Baseline RMS in ADC counts. data = baseline - data_orig", "baseline_rms"),
            "f4",
        ),
        (("Waveform data in raw ADC counts with 0 padding", "data"), "f4", (record_length_padded,)),
    ]

    records = np.zeros(len(raw_records), dtype=_dtype)
    strax.copy_to_buffer(raw_records, records, "_rr_to_r_led")

    mask = np.where((records["record_i"] == 0) & (records["length"] == led_cal_record_length))[0]
    records = records[mask]
    bl = records["data"][:, baseline_window[0] : baseline_window[1]].mean(axis=1)
    rms = records["data"][:, baseline_window[0] : baseline_window[1]].std(axis=1)
    records["data"][:, :led_cal_record_length] = (
        -1.0 * (records["data"][:, :led_cal_record_length].transpose() - bl[:]).transpose()
    )
    records["baseline"] = bl
    records["baseline_rms"] = rms
    return records


def get_led_windows(
    records,
    default_window,
    led_hit_extension,
    hit_min_height_over_noise,
    record_length,
    area_averaging_length,
):
    """Search for hits in the records, if a hit is found, return an interval around the hit given by
    led_hit_extension. If no hit is found in the record, return the default window.

    :param records: Array of the records to search for LED hits.
    :param default_window: Default window to use if no LED hit is found given as a tuple (start,
        end)
    :param led_hit_extension: The integration window around the first hit found to use. A tuple of
        form (samples_before, samples_after) the first LED hit.
    :param hit_min_amplitude: Minimum amplitude of the signal to be considered a hit.
    :param hit_min_height_over_noise: Minimum height of the signal over noise to be considered a
        hit. :return (len(records), 2) array: Integration window for each record
    :param record_length: The length of one led_calibration record
    :param area_averaging_length: The length (samples) of the window to do the averaging on.

    """
    hits = strax.find_hits(
        records,
        min_amplitude=0,  # Always use the height over noise threshold.
        min_height_over_noise=hit_min_height_over_noise,
    )
    # Check if the records are sorted properly by 'record_i' first and 'time' second and if them if
    # they are not
    record_i = hits["record_i"]
    time = hits["time"]
    if not (
        np.all(
            (record_i[:-1] < record_i[1:])
            | ((record_i[:-1] == record_i[1:]) & (time[:-1] <= time[1:]))
        )
    ):
        hits.sort(order=["record_i", "time"])

    default_windows = np.tile(default_window, (len(records), 1))
    return _get_led_windows(
        hits, default_windows, led_hit_extension, record_length, area_averaging_length
    )


@numba.jit(nopython=True)
def _get_led_windows(
    hits, default_windows, led_hit_extension, record_length, area_averaging_length
):
    windows = default_windows
    if len(default_windows) == 0:
        return np.empty((0, 2), dtype=np.int64)  # Return empty array of correct shape
    hit_left_min = default_windows[0, 0] - led_hit_extension[0]
    hit_left_max = record_length - area_averaging_length - led_hit_extension[1]
    last = -1

    for hit in hits:
        if hit["record_i"] == last:
            continue  # If there are multiple hits in one record, ignore after the first

        hit_left = hit["left"]
        # Limit the position of the window so it stays inside the record.
        if hit_left < hit_left_min:
            hit_left = hit_left_min
        if hit_left > hit_left_max:
            hit_left = hit_left_max

        left = hit_left + led_hit_extension[0]
        right = hit_left + led_hit_extension[1]

        windows[hit["record_i"]] = np.array([left, right])
        last = hit["record_i"]

    return windows


_on_off_dtype = np.dtype([("channel", "int16"), ("amplitude", "float32")])


@numba.jit(nopython=True)
def get_amplitude(records, led_windows, noise_window):
    """Needed for the SPE computation.

    Get the maximum of the signal in two different regions, one where there is no signal, and one
    where there is.

    :param records: Array of records
    :param ndarray led_windows : 2d array of shape (len(records), 2) with the window to use as the
        signal on area for each record. Inclusive left boundary and exclusive right boundary.
    :param tuple noise_window: Tuple with the window, used for the signal off area for all records.
    :return ndarray ons: 1d array of length len(records). The maximum amplitude in the led window
        area for each record.
    :return ndarray offs: 1d array of length len(records). The maximum amplitude in the noise area
        for each record.

    """
    ons = np.zeros(len(records), dtype=_on_off_dtype)
    offs = np.zeros(len(records), dtype=_on_off_dtype)

    for i, record in enumerate(records):
        ons[i]["channel"] = record["channel"]
        offs[i]["channel"] = record["channel"]

        ons[i]["amplitude"] = np.max(record["data"][led_windows[i, 0] : led_windows[i, 1]])
        offs[i]["amplitude"] = np.max(record["data"][noise_window[0] : noise_window[1]])

    return ons, offs


_area_dtype = np.dtype([("channel", "int16"), ("area", "float32")])


@numba.jit(nopython=True)
def get_area(records, led_windows, area_averaging_length, area_averaging_step):
    """Needed for the gain computation.

    Integrate the record in the defined window area. To reduce the effects of the noise, this is
    done with 6 different window lengths, which are then averaged.

    :param records: Array of records
    :param ndarray led_windows : 2d array of shape (len(records), 2) with the window to use as the
        integration boundaries.
    :param area_averaging_length: The total length in records of the window over which to do the
        averaging of the areas.
    :param area_averaging_step: The increase in length for each step of the averaging.
    :return ndarray area: 1d array of length len(records) with the averaged integrated areas for
        each record.

    """
    area = np.zeros(len(records), dtype=_area_dtype)
    end_pos = np.arange(0, area_averaging_length + area_averaging_step, area_averaging_step)

    for i, record in enumerate(records):
        area[i]["channel"] = record["channel"]
        for right in end_pos:
            area[i]["area"] += np.sum(record["data"][led_windows[i, 0] : led_windows[i, 1] + right])

        area[i]["area"] /= float(len(end_pos))

    return area


@export
class nVetoExtTimings(strax.Plugin):
    """Plugin which computes the time difference `delta_time` from pulse timing of `hitlets_nv` to
    start time of `raw_records` which belong the `hitlets_nv`.

    They are used as the external trigger timings.

    """

    __version__ = "0.0.1"

    depends_on = ("raw_records_nv", "hitlets_nv")
    provides = "ext_timings_nv"
    data_kind = "hitlets_nv"

    compressor = "zstd"

    channel_map = straxen.URLConfig(
        track=False,
        type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) channel number.",
    )

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_dt_fields
        dtype += [
            (("Delta time from trigger timing [ns]", "delta_time"), np.int16),
            (
                ("Index to which pulse (not record) the hitlet belongs to.", "pulse_i"),
                np.int32,
            ),
        ]
        return dtype

    def setup(self):
        self.nv_pmt_start = self.channel_map["nveto"][0]
        self.nv_pmt_stop = self.channel_map["nveto"][1] + 1

    def compute(self, hitlets_nv, raw_records_nv):
        rr_nv = raw_records_nv[raw_records_nv["record_i"] == 0]
        pulses = np.zeros(len(rr_nv), dtype=self.pulse_dtype())
        pulses["time"] = rr_nv["time"]
        pulses["endtime"] = rr_nv["time"] + rr_nv["pulse_length"] * rr_nv["dt"]
        pulses["channel"] = rr_nv["channel"]

        ext_timings_nv = np.zeros_like(hitlets_nv, dtype=self.dtype)
        ext_timings_nv["time"] = hitlets_nv["time"]
        ext_timings_nv["length"] = hitlets_nv["length"]
        ext_timings_nv["dt"] = hitlets_nv["dt"]
        self.calc_delta_time(
            ext_timings_nv, pulses, hitlets_nv, self.nv_pmt_start, self.nv_pmt_stop
        )

        return ext_timings_nv

    @staticmethod
    def pulse_dtype():
        pulse_dtype = []
        pulse_dtype += strax.time_fields
        pulse_dtype += [(("PMT channel", "channel"), np.int16)]
        return pulse_dtype

    @staticmethod
    def calc_delta_time(ext_timings_nv_delta_time, pulses, hitlets_nv, nv_pmt_start, nv_pmt_stop):
        """Numpy access with fancy index returns copy, not view This for-loop is required to
        substitute in one by one."""
        hitlet_index = np.arange(len(hitlets_nv))
        pulse_index = np.arange(len(pulses))
        for ch in range(nv_pmt_start, nv_pmt_stop):
            mask_hitlets_in_channel = hitlets_nv["channel"] == ch
            hitlet_in_channel_index = hitlet_index[mask_hitlets_in_channel]

            mask_pulse_in_channel = pulses["channel"] == ch
            pulse_in_channel_index = pulse_index[mask_pulse_in_channel]

            hitlets_in_channel = hitlets_nv[hitlet_in_channel_index]
            pulses_in_channel = pulses[pulse_in_channel_index]
            hit_in_pulse_index = strax.fully_contained_in(hitlets_in_channel, pulses_in_channel)
            for h_i, p_i in zip(hitlet_in_channel_index, hit_in_pulse_index):
                if p_i == -1:
                    continue
                res = ext_timings_nv_delta_time[h_i]

                res["delta_time"] = (
                    hitlets_nv[h_i]["time"]
                    + hitlets_nv[h_i]["time_amplitude"]
                    - pulses_in_channel[p_i]["time"]
                )
                res["pulse_i"] = pulse_in_channel_index[p_i]
