import numpy as np
import numba
import strax
import straxen


export, __all__ = strax.exporter()


@export
class PeakAmbience(strax.OverlapWindowPlugin):
    """Calculate Ambience of peaks. Features are the number of lonehits, small S0, S1, S2 in a time
    window before peaks, and the number of small S2 in circle near the S2 peak in a time window.

    References:
        * v0.0.7 reference: xenon:xenonnt:ac:prediction:shadow_ambience
        * v0.1.0 reference: xenon:xenonnt:analysis:redefine_n_competing

    """

    __version__ = "0.1.0"
    depends_on = ("peak_basics", "peak_positions", "lone_hits")
    provides = "peak_ambience"
    data_kind = "peaks"
    save_when = strax.SaveWhen.EXPLICIT

    gain_model = straxen.URLConfig(
        infer_type=False, help="PMT gain model. Specify as URL or explicit value"
    )

    ambience_time_window_backward = straxen.URLConfig(
        default=int(2e6), type=int, track=True, help="Search for ambience in this time window [ns]"
    )

    ambience_divide_t = straxen.URLConfig(
        default=True,
        type=bool,
        track=True,
        help="Whether to divide area by time difference of ambience creating peak to current peak",
    )

    ambience_divide_r = straxen.URLConfig(
        default=False,
        type=bool,
        track=True,
        help="Whether to divide area by radial distance of ambience creating peak to current peak",
    )

    ambient_radius = straxen.URLConfig(
        default=6.7, type=float, track=True, help="Search for ambience in this radius [cm]"
    )

    ambience_area_parameters = straxen.URLConfig(
        default=(30, 30, 60),
        type=(list, tuple),
        track=True,
        help="The upper limit of S0, S1, S2 area to be counted",
    )

    def get_window_size(self):
        return (10 * self.ambience_time_window_backward, 0)

    @property
    def origin_dtype(self):
        return ["lh_before", "s0_before", "s1_before", "s2_before", "s2_near"]

    def infer_dtype(self):
        dtype = []
        for ambience in self.origin_dtype:
            dtype += [
                (
                    (f"Number of small {' '.join(ambience.split('_'))} a peak", f"n_{ambience}"),
                    np.int16,
                ),
                (
                    (f"Sum of small {' '.join(ambience.split('_'))} a peak", f"s_{ambience}"),
                    np.float32,
                ),
            ]
        dtype += [
            (
                ("Sum of small hits and peaks before a peak", "s_before"),
                np.float32,
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def setup(self):
        self.to_pe = self.gain_model

    def compute(self, lone_hits, peaks):
        argsort = strax.stable_argsort(peaks["center_time"])
        _peaks = peaks[argsort].copy()
        result = np.zeros(len(peaks), self.dtype)
        _quick_assign(argsort, result, self.compute_ambience(lone_hits, peaks, _peaks))
        return result

    def compute_ambience(self, lone_hits, peaks, current_peak):
        # 1. Initialization
        result = np.zeros(len(current_peak), self.dtype)

        # 2. Define time window for each peak,
        # we will find small peaks & lone hits within these time windows
        roi = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi["time"] = current_peak["center_time"] - self.ambience_time_window_backward
        roi["endtime"] = current_peak["center_time"]

        # 3. Calculate number and area sum of lonehits before a peak
        touching_windows = strax.touching_windows(lone_hits, roi)
        # Calculating ambience
        self.lonehits_ambience(
            current_peak,
            lone_hits,
            touching_windows,
            result["n_lh_before"],
            result["s_lh_before"],
            self.ambience_divide_t,
            self.to_pe,
        )

        # 4. Calculate number and area sum of small S0, S1, S2 before a peak
        radius = -1
        for stype, area in zip([0, 1, 2], self.ambience_area_parameters):
            mask_pre = (peaks["type"] == stype) & (peaks["area"] < area)
            touching_windows = strax.touching_windows(peaks[mask_pre], roi)
            # Calculating ambience
            self.peaks_ambience(
                current_peak,
                peaks[mask_pre],
                touching_windows,
                radius,
                result[f"n_s{stype}_before"],
                result[f"s_s{stype}_before"],
                self.ambience_divide_t,
                self.ambience_divide_r,
            )

        # 5. Calculate number and area sum of small S2 near(in (x,y) space) a S2 peak
        mask_pre = (peaks["type"] == 2) & (peaks["area"] < self.ambience_area_parameters[2])
        touching_windows = strax.touching_windows(peaks[mask_pre], roi)
        # Calculating ambience
        self.peaks_ambience(
            current_peak,
            peaks[mask_pre],
            touching_windows,
            self.ambient_radius,
            result["n_s2_near"],
            result["s_s2_near"],
            self.ambience_divide_t,
            self.ambience_divide_r,
        )

        # 6. Set time and endtime for peaks
        result["time"] = current_peak["time"]
        result["endtime"] = strax.endtime(current_peak)

        # 7. Calculate sum of small hits and peaks before a peak
        result["s_before"] = (
            result["s_lh_before"]
            + result["s_s0_before"]
            + result["s_s1_before"]
            + result["s_s2_before"]
        )
        return result

    @staticmethod
    @numba.njit
    def lonehits_ambience(
        peaks, pre_hits, touching_windows, num_array, sum_array, ambience_divide_t, to_pe
    ):
        # Function to find lonehits before a peak
        # creating_hit is the lonehit creating ambience
        # suspicious_peak is the suspicious peak in the ambience created by creating_hit
        for p_i, suspicious_peak in enumerate(peaks):
            indices = touching_windows[p_i]
            for idx in range(indices[0], indices[1]):
                creating_hit = pre_hits[idx]
                dt = suspicious_peak["center_time"] - creating_hit["time"]
                if (dt <= 0) or (creating_hit["area"] <= 0):
                    continue
                num_array[p_i] += 1
                # Sometimes we may interested in sum of area / dt
                if ambience_divide_t:
                    sum_array[p_i] += creating_hit["area"] * to_pe[creating_hit["channel"]] / dt
                else:
                    sum_array[p_i] += creating_hit["area"]

    @staticmethod
    @numba.njit
    def peaks_ambience(
        peaks,
        pre_peaks,
        touching_windows,
        ambient_radius,
        num_array,
        sum_array,
        ambience_divide_t,
        ambience_divide_r,
    ):
        # Function to find S0, S1, S2 before or near a peak
        # creating_peak is the peak creating ambience
        # suspicious_peak is the suspicious peak in the ambience created by creating_peak
        for p_i, suspicious_peak in enumerate(peaks):
            indices = touching_windows[p_i]
            for idx in range(indices[0], indices[1]):
                creating_peak = pre_peaks[idx]
                r = distance_in_xy(suspicious_peak, creating_peak)
                dt = suspicious_peak["center_time"] - creating_peak["center_time"]
                if dt <= 0:
                    continue
                if (ambient_radius < 0) or (r <= ambient_radius):
                    num_array[p_i] += 1
                    # Sometimes we may interested in sum of area / dt
                    if ambience_divide_t:
                        sum_array[p_i] += creating_peak["area"] / dt
                    else:
                        sum_array[p_i] += creating_peak["area"]
                    # Sometimes we may interested in sum of area / r^2
                    if ambience_divide_r and ambient_radius > 0:
                        sum_array[p_i] /= r**2


@numba.njit
def distance_in_xy(peak_a, peak_b):
    """Distance between S2s in (x,y)"""
    return np.sqrt((peak_a["x"] - peak_b["x"]) ** 2 + (peak_a["y"] - peak_b["y"]) ** 2)


@numba.njit
def _quick_assign(indices, results, inputs):
    for i, r in zip(indices, inputs):
        results[i] = r
