import numpy as np
import numba
import strax
import straxen

from straxen.plugins.defaults import DEFAULT_POSREC_ALGO, FAR_XYPOS_S2_TYPE, WIDE_XYPOS_S2_TYPE
from .peak_proximity import half_cauchy_pdf
from .peak_ambience import distance_in_xy, _quick_assign

export, __all__ = strax.exporter()


@export
class PeakShadow(strax.OverlapWindowPlugin):
    """This plugin can find and calculate the time & position shadow from previous peaks in time. It
    also gives the area and (x, y) of the previous peaks.

    References:
        * v0.1.5 reference: xenon:xenonnt:ac:prediction:shadow_ambience

    """

    __version__ = "0.2.0"

    depends_on = ("peak_basics", "peak_positions")
    provides = "peak_shadow"
    save_when = strax.SaveWhen.EXPLICIT

    shadow_threshold = straxen.URLConfig(
        default={"s1_time_shadow": 1e3, "s2_time_shadow": 1e4, "s2_position_shadow": 1e4},
        type=dict,
        track=True,
        help="Only take S1/S2s larger than this into account when calculating Shadow [PE]",
    )

    shadow_sum = straxen.URLConfig(
        default={"s1_time_shadow": False, "s2_time_shadow": False, "s2_position_shadow": False},
        type=dict,
        track=True,
        help="Whether the shadow should be summed up rather than taking the largest one",
    )

    shadow_merge_replace = straxen.URLConfig(
        default={"s1_time_shadow": False, "s2_time_shadow": False, "s2_position_shadow": False},
        type=dict,
        track=True,
        help="Whether replace the primary peak with the merged peak",
    )

    shadow_merge_dt = straxen.URLConfig(
        default={"s1_time_shadow": -1, "s2_time_shadow": -1, "s2_position_shadow": -1},
        type=dict,
        track=True,
        help="Merge primary peaks within this time window [ns]",
    )

    shadow_merge_dr = straxen.URLConfig(
        default={"s1_time_shadow": -1, "s2_time_shadow": -1, "s2_position_shadow": -1},
        type=dict,
        track=True,
        help="Merge primary peaks within this distance [cm]",
    )

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )

    shadow_time_window_backward = straxen.URLConfig(
        default=int(1e9),
        type=int,
        track=True,
        help="Search for peaks casting time & position shadow in this time window [ns]",
    )

    shadow_uncertainty_weights = straxen.URLConfig(
        default=True,
        type=bool,
        track=True,
        help="Whether to use uncertainty weights when merging peaks",
    )

    shadow_deltatime_exponent = straxen.URLConfig(
        default=-1.0, type=float, track=True, help="The exponent of delta t when calculating shadow"
    )

    shadow_sigma_and_baseline = straxen.URLConfig(
        default=[15.220, 0.036],
        type=list,
        track=True,
        help=(
            "Fitted position correlation sigma[cm*PE^0.5] and baseline[cm] using in position shadow"
        ),
    )

    def get_window_size(self):
        return (10 * self.shadow_time_window_backward, 0)

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
                        f"Previous large {type_str} casted largest {tp_desc} shadow [PE/ns]",
                        f"shadow_{key}",
                    ),
                    np.float32,
                )
            )
            dtype.append(
                (
                    (
                        (
                            f"Time difference to the previous large {type_str} peak casting largest"
                            f" {tp_desc} shadow [ns]"
                        ),
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
                            f"X of previous large s2 peak casting largest {tp_desc} shadow [cm]",
                            f"x_{key}",
                        ),
                        np.float32,
                    )
                )
                dtype.append(
                    (
                        (
                            f"Y of previous large s2 peak casting largest {tp_desc} shadow [cm]",
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
                            f"The nearest previous large {type_str} [ns]",
                            f"nearest_{type_str}",
                        ),
                        np.float32,
                    )
                )
                dtype.append(
                    (
                        (
                            f"Time difference to the nearest previous large {type_str} [ns]",
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

    def setup(self):
        super().setup()
        for key in ["s2_position_shadow", "s2_time_shadow", "s1_time_shadow"]:
            if self.shadow_sum[key] and not self.shadow_merge_replace[key]:
                raise ValueError(
                    f"shadow_sum is set to True, but shadow_merge_replace is set to False for {key}"
                )
        if (
            self.shadow_uncertainty_weights
            and self.default_reconstruction_algorithm != DEFAULT_POSREC_ALGO
        ):
            raise ValueError(
                "Using pos-rec uncertainty as (x, y) merging weights, "
                f"but default_reconstruction_algorithm is not set to {DEFAULT_POSREC_ALGO}"
            )

    @staticmethod
    def cluster_xy(peaks, indices, dr):
        """Find connectivity map of peaks in xy space."""
        clusters = []
        unvisited = set(indices)

        while unvisited:
            current = unvisited.pop()
            cluster = {current}
            to_check = {current}
            while to_check:
                checking = to_check.pop()
                dists = distance_in_xy(peaks[checking], peaks[list(unvisited)])
                close = {i for i, d in zip(unvisited, dists) if d <= dr}
                to_check.update(close)
                cluster.update(close)
                unvisited -= close
            clusters.append(list(cluster))
        clusters = [sorted(cluster) for cluster in clusters if len(cluster) > 1]
        return clusters

    @staticmethod
    def merge_peaks(peaks, weights, dt, dr, replace=False):
        """Merge peaks in time and space."""

        # Prepare the dtype for the merged peaks
        dtype = []
        needed_fileds = "time endtime center_time area x y".split()
        for d, n in zip(peaks.dtype.descr, peaks.dtype.names):
            if n in needed_fileds:
                dtype.append(d)
        dtype = np.dtype(dtype)

        # Copy the fields to a new array
        _peaks = np.zeros(len(peaks), dtype=dtype)
        for d in needed_fileds:
            _peaks[d] = peaks[d]

        # Identify indices of peaks mergable in time
        if dt > 0:
            t_clusters = [[0]]
            for i in range(len(peaks) - 1):
                if np.abs(peaks["center_time"][i + 1] - peaks["center_time"][i]) > dt:
                    t_clusters.append([i])
                else:
                    t_clusters[-1].append(i + 1)
            t_clusters = [cluster for cluster in t_clusters if len(cluster) > 1]
        else:
            t_clusters = []

        # Identify indices of peaks mergable also in space
        if dr > 0:
            txy_clusters = []
            for cluster in t_clusters:
                txy_clusters += PeakShadow.cluster_xy(peaks, cluster, dr)
        else:
            txy_clusters = t_clusters

        # Merge peaks
        merged_peaks = np.zeros(len(txy_clusters), dtype=dtype)
        for i, cluster in enumerate(txy_clusters):
            # Merge peaks
            merged_peaks[i]["time"] = peaks["time"][cluster].min()
            merged_peaks[i]["endtime"] = strax.endtime(peaks)[cluster].max()
            # center_time is by definition the average time of waveform
            merged_peaks[i]["center_time"] = np.average(
                peaks["center_time"][cluster], weights=peaks["area"][cluster]
            )
            merged_peaks[i]["area"] = np.sum(peaks["area"][cluster])
            merged_peaks[i]["x"] = np.average(peaks["x"][cluster], weights=weights[cluster])
            merged_peaks[i]["y"] = np.average(peaks["y"][cluster], weights=weights[cluster])

        # Replace peaks with merged peaks if replace is True
        if replace:
            for i, cluster in enumerate(txy_clusters):
                _peaks[cluster] = merged_peaks[i]
            _peaks = np.delete(_peaks, [cluster[1:] for cluster in txy_clusters])
        else:
            _peaks = np.insert(_peaks, [cluster[0] for cluster in txy_clusters], merged_peaks)
        _peaks = strax.sort_by_time(_peaks)

        # Return merged peaks
        return _peaks

    @property
    def shadowdtype(self):
        dtype = []
        dtype += [("shadow", np.float32), ("dt", np.int64)]
        dtype += [("x", np.float32), ("y", np.float32)]
        dtype += [("nearest", np.float32), ("nearest_dt", np.int64)]
        return dtype

    def compute(self, peaks):
        argsort = strax.stable_argsort(peaks["center_time"])
        _peaks = peaks[argsort].copy()
        result = np.zeros(len(peaks), self.dtype)
        _quick_assign(argsort, result, self.compute_shadow(peaks, _peaks))
        return result

    def compute_shadow(self, peaks, current_peak):
        # 1. Define time window for each peak, we will find previous peaks within these time windows
        roi = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi["time"] = current_peak["center_time"] - self.shadow_time_window_backward
        roi["endtime"] = current_peak["center_time"]

        # 2. Calculate S2 position shadow, S2 time shadow, and S1 time shadow
        # weights of the peaklets when calculating the weighted mean deviation in (x, y)
        posrec_algo = self.default_reconstruction_algorithm
        if self.shadow_uncertainty_weights:
            weights = 1 / peaks[f"position_contour_area_{posrec_algo}"]
        else:
            weights = peaks["area"] * peaks["area_fraction_top"]
        result = np.zeros(len(current_peak), self.dtype)
        for key in ["s2_position_shadow", "s2_time_shadow", "s1_time_shadow"]:
            type_str = key.split("_")[0]
            if "s2" in key:
                # in very rare cases, high energy S2 canbe classified as type 20 and 22
                stype = [2, FAR_XYPOS_S2_TYPE, WIDE_XYPOS_S2_TYPE]
            else:
                # type 3 is also S1 xenon:xenonnt:lsanchez:som_sr1b_summary_note
                stype = [1, 3]
            mask_pre = np.isin(peaks["type"], stype) & (peaks["area"] > self.shadow_threshold[key])
            _peaks = self.merge_peaks(
                peaks[mask_pre],
                weights[mask_pre],
                dt=self.shadow_merge_dt[key],
                dr=self.shadow_merge_dr[key],
                replace=self.shadow_merge_replace[key],
            )
            split_peaks = strax.touching_windows(_peaks, roi)
            array = np.zeros(len(current_peak), np.dtype(self.shadowdtype))
            strax.set_nan_defaults(array)

            # Initialization
            array["dt"] = self.shadow_time_window_backward
            array["nearest_dt"] = self.shadow_time_window_backward
            # The default value for shadow is set to be the lowest possible value
            if "time" in key:
                array["shadow"] = (
                    self.shadow_threshold[key] * array["dt"] ** self.shadow_deltatime_exponent
                )
            else:
                # Assign the position shadow to be zero
                array["shadow"] = 0

            # Calculating shadow, the Major of the plugin.
            # Only record the previous peak casting the largest shadow if is_sum is False
            is_position = "position" in key
            is_sum = self.shadow_sum[key]
            if is_position and is_sum:
                raise ValueError(
                    "Position shadow cannot be summed up, please set its shadow_sum to False"
                )
            if len(current_peak):
                self.peaks_shadow(
                    current_peak,
                    _peaks,
                    split_peaks,
                    self.shadow_deltatime_exponent,
                    array,
                    is_position,
                    is_sum,
                    self.getsigma(self.shadow_sigma_and_baseline, current_peak["area"]),
                )

            # Fill results
            names = ["shadow", "dt"]
            if "s2" in key:  # Only previous S2 peaks have (x,y)
                names += ["x", "y"]
            if "time" in key:  # Only time shadow gives the nearest large peak
                names += ["nearest", "nearest_dt"]
            for name in names:
                if "nearest" in name:
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
        result["pdf_s2_position_shadow"] = half_cauchy_pdf(
            distance,
            self.getsigma(self.shadow_sigma_and_baseline, current_peak["area"]),
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
    def peaks_shadow(
        peaks, pre_peaks, touching_windows, exponent, result, is_position, is_sum, sigmas=None
    ):
        """For each peak in peaks, check if there is a shadow-casting peak and check if it casts the
        largest shadow."""
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
                if dt < result["nearest_dt"][p_i]:
                    result["nearest_dt"][p_i] = dt
                    result["nearest"][p_i] = casting_peak["area"]
                # Calculate time shadow
                new_shadow = casting_peak["area"] * dt**exponent
                if is_position:
                    # Calculate position shadow which is
                    # time shadow with a HalfCauchy PDF multiplier
                    distance = distance_in_xy(suspicious_peak, casting_peak)
                    distance = np.where(np.isnan(distance), 2 * straxen.tpc_r, distance)
                    new_shadow *= half_cauchy_pdf(distance, sigma).item()
                # Only the previous peak with largest shadow is recorded
                if new_shadow > result["shadow"][p_i]:
                    result["shadow"][p_i] = new_shadow
                    result["x"][p_i] = casting_peak["x"]
                    result["y"][p_i] = casting_peak["y"]
                    result["dt"][p_i] = suspicious_peak["center_time"] - casting_peak["center_time"]
            if is_sum:
                sl = slice(indices[0], indices[1])
                dt = suspicious_peak["center_time"] - pre_peaks["center_time"][sl]
                mask = dt > 0
                if mask.sum() == 0:
                    continue
                ft = dt[mask] ** exponent
                result["shadow"][p_i] = np.sum(pre_peaks["area"][sl][mask] * ft)
                result["x"][p_i] = np.nan
                result["y"][p_i] = np.nan
                result["dt"][p_i] = mask.sum() / np.sum(ft) ** (1 / exponent)
