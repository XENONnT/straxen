"""
Dear nT analyser,
if you want to complain please contact:
    chiara@physik.uzh.ch, gvolta@physik.uzh.ch, kazama@isee.nagoya-u.ac.jp
"""

from immutabledict import immutabledict
import strax
import straxen
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

    baseline_window = straxen.URLConfig(
        default=(0, 40), infer_type=False, help="Window (samples) for baseline calculation."
    )

    default_led_window = straxen.URLConfig(
        default=(78, 132),
        infer_type=False,
        help="Default window (samples) to integrate and get he maximum amplitude of if no hit was found in the record.",
    )

    led_hit_extension = straxen.URLConfig(
        default=(-10, 44),
        infer_type=False,
        help="The extension around the LED hit to integrate."
    )

    noise_window = straxen.URLConfig(
        default=(10, 48), 
        infer_type=False, 
        help="Window (samples) to analyse the noise"
    )

    channel_list = straxen.URLConfig(
        default=(tuple(channel_list)),
        infer_type=False,
        help="List of PMTs. Defalt value: all the PMTs",
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

        This is used for the different ligh levels. As defaul value all the PMTs are considered.

        """
        mask = np.where(np.in1d(raw_records["channel"], self.channel_list))[0]
        rr = raw_records[mask]
        r = get_records(rr, baseline_window=self.baseline_window)
        del rr, raw_records

        temp = np.zeros(len(r), dtype=self.dtype)
        strax.copy_to_buffer(r, temp, "_recs_to_temp_led")
        
        led_windows = get_led_windows(r, self.led_window, self.led_hit_extension)

        on, off = get_amplitude(r, led_windows, self.noise_window)
        temp["amplitude_led"] = on["amplitude"]
        temp["amplitude_noise"] = off["amplitude"]

        area = get_area(r, led_windows)
        temp["area"] = area["area"]
        return temp


def get_records(raw_records, baseline_window):
    """Determine baseline as the average of the first baseline_samples of each pulse.

    Subtract the pulse float(data) from baseline.

    """

    record_length = np.shape(raw_records.dtype["data"])[0]

    _dtype = [
        (("Start time since unix epoch [ns]", "time"), "<i8"),
        (("Length of the interval in samples", "length"), "<i4"),
        (("Width of one sample [ns]", "dt"), "<i2"),
        (("Channel/PMT number", "channel"), "<i2"),
        (
            ("Length of pulse to which the record belongs (without zero-padding)", "pulse_length"),
            "<i4",
        ),
        (("Fragment number in the pulse", "record_i"), "<i2"),
        (("Waveform data in raw ADC counts", "data"), "f4", (record_length,)),
    ]

    records = np.zeros(len(raw_records), dtype=_dtype)
    strax.copy_to_buffer(raw_records, records, "_rr_to_r_led")

    mask = np.where((records["record_i"] == 0) & (records["length"] == 160))[0]
    records = records[mask]
    bl = records["data"][:, baseline_window[0] : baseline_window[1]].mean(axis=1)
    records["data"][:, :160] = -1.0 * (records["data"][:, :160].transpose() - bl[:]).transpose()
    return records


def get_led_windows(records, default_window, led_hit_extension):
    """ Searches for hits in the records, if a hit is found, returns an interval
        around the hit given by led_hit_extension. If no hit is found in the
        record, returns the default window.

    :return (len(records), 2) array: Integration window for each record

    """    
    hits = strax.find_hits(records)
    default_windows = np.tile(default_window, (len(records), 1))
    return _get_led_windows(hits, default_windows, led_hit_extension)


@numba.jit(nopython=True)
def _get_led_windows(hits, default_windows, led_hit_extension):
    """ Searches for hits in the records, if a hit is found, returns an interval
        around the hit given by led_hit_extension. If no hit is found in the
        record, returns the default window.

    :return (len(records), 2) array: Integration window for each record

    """    
    windows = default_windows
    # max_window = np.array([96, 150])
    last = -1
    for hit in hits:
        if hit["record_i"] == last:
            continue # If there are multiple hits in one record, ignore after the first

        left = hit["left"] + led_hit_extension[0]
        # Limit the position of the window so it stays inside the record.
        if left < default_windows[0, 0]: left = default_windows[0, 0]
        elif left > 96: left = 96

        right = hit["left"] + led_hit_extension[1]
        if right < default_windows[0, 1]: right = default_windows[0, 1]
        elif right > 150: right = 150

        windows[hit["record_i"]] = np.array([left, right])
        last = hit["record_i"]
        
    return windows


_on_off_dtype = np.dtype([("channel", "int16"), ("amplitude", "float32")])


@numba.jit(nopython=True)
def get_amplitude(records, led_windows, noise_window):
    """Needed for the SPE computation.

    Take the maximum in two different regions, where there is the signal and where there is not.

    """
    ons = np.zeros(len(records), dtype=_on_off_dtype)
    offs = np.zeros(len(records), dtype=_on_off_dtype)

    for i, record in enumerate(records):
        ons[i]["channel"] = record["channel"]
        offs[i]["channel"] = record["channel"]

        ons[i]["amplitude"] = np.max(record["data"][led_windows[i, 0]:led_windows[i, 1]+1])
        offs[i]["amplitude"] = np.max(record["data"][noise_window[0]:noise_window[1]+1])

    return ons, offs


_area_dtype = np.dtype([("channel", "int16"), ("area", "float32")])


@numba.jit(nopython=True)
def get_area(records, led_windows):
    """Needed for the gain computation.

    Sum the data in the defined window to get the area. This is done in 6 integration window and it
    returns the average area.

    """
    area = np.zeros(len(records), dtype=_area_dtype)
    end_pos = [2 * i for i in range(6)]

    for i, record in enumerate(records):
        area[i]["channel"] = record["channel"]
        for right in end_pos:
            area[i]["area"] += np.sum(record["data"][led_windows[i, 0]:led_windows[i, 1]+right+1])

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
            (("Index to which pulse (not record) the hitlet belongs to.", "pulse_i"), np.int32),
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
