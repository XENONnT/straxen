import numpy as np
import numba
from scipy.stats import halfcauchy
from .peak_ambience import distance_in_xy, _quick_assign
import strax
import straxen

export, __all__ = strax.exporter()


@export
class PeakShadow(strax.OverlapWindowPlugin):
    """This plugin can find and calculate the time & position shadow from
    previous peaks in time. It also gives the area and (x,y) of the previous
    peaks.

    References:
        * v0.1.5 reference: xenon:xenonnt:ac:prediction:shadow_ambience
    """

    __version__ = "0.1.6"

    depends_on = ("peak_basics", "peak_positions")
    provides = "peak_shadow"
    save_when = strax.SaveWhen.EXPLICIT

    shadow_time_window_backward = straxen.URLConfig(
        default=int(1e9),
        type=int,
        track=True,
        help="Search for peaks casting time & position shadow in this time window [ns]",
    )

    shadow_threshold = straxen.URLConfig(
        default={"s1_time_shadow": 1e3, "s2_time_shadow": 1e4, "s2_position_shadow": 1e4},
        type=dict,
        track=True,
        help="Only take S1/S2s larger than this into account when calculating Shadow [PE]",
    )

    shadow_deltatime_exponent = straxen.URLConfig(
        default=-1.0, type=float, track=True, help="The exponent of delta t when calculating shadow"
    )

    shadow_sigma_and_baseline = straxen.URLConfig(
        default=[15.220, 0.036],
        type=list,
        track=True,
        help="Fitted position correlation sigma[cm*PE^0.5] and baseline[cm] using in position shadow",
    )

    def get_window_size(self):
        return 10 * self.shadow_time_window_backward

    def infer_dtype(self):
        s1_time_shadow_dtype = []
        s2_time_shadow_dtype = []
        s2_position_shadow_dtype = []
        nearest_dtype = []
        # We have time shadow(S2/dt) and position shadow(S2/dt*p(s))
        # previous S1 can only cast time shadow, previous S2 can cast both time & position shadow
        for key, dtype in zip(
            ["s1_time_shadow", "s2_time_shadow", "s2_position_shadow"],
            [s1_time_shadow_dtype, s2_time_shadow_dtype, s2_position_shadow_dtype],
        ):
            type_str, tp_desc, _ = key.split("_")
            dtype.append(
                (
                    (
                        f"previous large {type_str} casted largest {tp_desc} shadow [PE/ns]",
                        f"shadow_{key}",
                    ),
                    np.float32,
                )
            )
            dtype.append(
                (
                    (
                        f"time difference to the previous large {type_str} peak casting largest {tp_desc} shadow [ns]",
                        f"dt_{key}",
                    ),
                    np.int64,
                )
            )
            # Only previous S2 peaks have (x,y)
            if "s2" in key:
                dtype.append(
                    (
                        (
                            f"x of previous large s2 peak casting largest {tp_desc} shadow [cm]",
                            f"x_{key}",
                        ),
                        np.float32,
                    )
                )
                dtype.append(
                    (
                        (
                            f"y of previous large s2 peak casting largest {tp_desc} shadow [cm]",
                            f"y_{key}",
                        ),
                        np.float32,
                    )
                )
            # Only time shadow gives the nearest large peak
            if "time" in key:
                dtype.append(
                    (
                        (
                            f"time difference to the nearest previous large {type_str} [ns]",
                            f"nearest_dt_{type_str}",
                        ),
                        np.int64,
                    )
                )
        # Also record the PDF of HalfCauchy when calculating S2 position shadow
        s2_position_shadow_dtype.append(
            (
                ("PDF describing correlation to the previous large s2", "pdf_s2_position_shadow"),
                np.float32,
            )
        )

        dtype = (
            s1_time_shadow_dtype
            + s2_time_shadow_dtype
            + s2_position_shadow_dtype
            + nearest_dtype
            + strax.time_fields
        )
        return dtype

    @property
    def shadowdtype(self):
        dtype = []
        dtype += [("shadow", np.float32), ("dt", np.int64)]
        dtype += [("x", np.float32), ("y", np.float32)]
        dtype += [("nearest_dt", np.int64)]
        return dtype

    def compute(self, peaks):
        argsort = np.argsort(peaks["center_time"], kind="mergesort")
        _peaks = np.sort(peaks, order="center_time")
        result = np.zeros(len(peaks), self.dtype)
        _quick_assign(argsort, result, self.compute_shadow(peaks, _peaks))
        return result

    def compute_shadow(self, peaks, current_peak):
        # 1. Define time window for each peak, we will find previous peaks within these time windows
        roi_shadow = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi_shadow["time"] = current_peak["center_time"] - self.shadow_time_window_backward
        roi_shadow["endtime"] = current_peak["center_time"]

        # 2. Calculate S2 position shadow, S2 time shadow, and S1 time shadow
        result = np.zeros(len(current_peak), self.dtype)
        for key in ["s2_position_shadow", "s2_time_shadow", "s1_time_shadow"]:
            is_position = "position" in key
            type_str = key.split("_")[0]
            stype = 2 if "s2" in key else 1
            mask_pre = (peaks["type"] == stype) & (peaks["area"] > self.shadow_threshold[key])
            split_peaks = strax.touching_windows(peaks[mask_pre], roi_shadow)
            array = np.zeros(len(current_peak), np.dtype(self.shadowdtype))

            # Initialization
            array["x"] = np.nan
            array["y"] = np.nan
            array["dt"] = self.shadow_time_window_backward
            # The default value for shadow is set to be the lowest possible value
            if "time" in key:
                array["shadow"] = (
                    self.shadow_threshold[key] * array["dt"] ** self.shadow_deltatime_exponent
                )
            else:
                array["shadow"] = 0
            array["nearest_dt"] = self.shadow_time_window_backward

            # Calculating shadow, the Major of the plugin. Only record the previous peak casting the largest shadow
            if len(current_peak):
                self.peaks_shadow(
                    current_peak,
                    peaks[mask_pre],
                    split_peaks,
                    self.shadow_deltatime_exponent,
                    array,
                    is_position,
                    self.getsigma(self.shadow_sigma_and_baseline, current_peak["area"]),
                )

            # Fill results
            names = ["shadow", "dt"]
            if "s2" in key:  # Only previous S2 peaks have (x,y)
                names += ["x", "y"]
            if "time" in key:  # Only time shadow gives the nearest large peak
                names += ["nearest_dt"]
            for name in names:
                if name == "nearest_dt":
                    result[f"{name}_{type_str}"] = array[name]
                else:
                    result[f"{name}_{key}"] = array[name]

        distance = np.sqrt(
            (result[f"x_s2_position_shadow"] - current_peak["x"]) ** 2
            + (result[f"y_s2_position_shadow"] - current_peak["y"]) ** 2
        )
        # If distance is NaN, set largest distance
        distance = np.where(np.isnan(distance), 2 * straxen.tpc_r, distance)
        # HalfCauchy PDF when calculating S2 position shadow
        result["pdf_s2_position_shadow"] = halfcauchy.pdf(
            distance, scale=self.getsigma(self.shadow_sigma_and_baseline, current_peak["area"])
        )

        # 6. Set time and endtime for peaks
        result["time"] = current_peak["time"]
        result["endtime"] = strax.endtime(current_peak)
        return result

    @staticmethod
    @np.errstate(invalid="ignore")
    def getsigma(sigma_and_baseline, s2):
        # The parameter of HalfCauchy, which is a function of S2 area
        return sigma_and_baseline[0] / np.sqrt(s2) + sigma_and_baseline[1]

    @staticmethod
    @numba.njit
    def peaks_shadow(peaks, pre_peaks, touching_windows, exponent, result, pos_corr, sigmas=None):
        """For each peak in peaks, check if there is a shadow-casting peak and
        check if it casts the largest shadow."""
        for p_i, (suspicious_peak, sigma) in enumerate(zip(peaks, sigmas)):
            # casting_peak is the previous large peak casting shadow
            # suspicious_peak is the suspicious peak which in shadow from casting_peak
            indices = touching_windows[p_i]
            for idx in range(indices[0], indices[1]):
                casting_peak = pre_peaks[idx]
                dt = suspicious_peak["center_time"] - casting_peak["center_time"]
                if dt <= 0:
                    continue
                # First we record the time difference to the nearest previous peak
                result["nearest_dt"][p_i] = min(result["nearest_dt"][p_i], dt)
                # Calculate time shadow
                new_shadow = casting_peak["area"] * dt**exponent
                if pos_corr:
                    # Calculate position shadow which is time shadow with a HalfCauchy PDF multiplier
                    distance = distance_in_xy(suspicious_peak, casting_peak)
                    distance = np.where(np.isnan(distance), 2 * straxen.tpc_r, distance)
                    new_shadow *= 2 / (np.pi * sigma * (1 + (distance / sigma) ** 2))
                # Only the previous peak with largest shadow is recorded
                if new_shadow > result["shadow"][p_i]:
                    result["shadow"][p_i] = new_shadow
                    result["x"][p_i] = casting_peak["x"]
                    result["y"][p_i] = casting_peak["y"]
                    result["dt"][p_i] = suspicious_peak["center_time"] - casting_peak["center_time"]
