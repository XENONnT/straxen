import numpy as np
import numba
import strax
from strax import parse_selection
import straxen
from straxen.plugins.peaks.peak_ambience import distance_in_xy, _quick_assign
from straxen.plugins.defaults import FAR_XYPOS_S2_TYPE, WIDE_XYPOS_S2_TYPE


class PeakAmbience_(strax.OverlapWindowPlugin):
    __version__ = "0.0.0"
    save_when = strax.SaveWhen.EXPLICIT
    depends_on = ("peak_basics", "peak_positions", "lone_hits")
    provides = "peak_ambience_"

    gain_model = straxen.URLConfig(
        infer_type=False, help="PMT gain model. Specify as URL or explicit value"
    )

    ambience_time_window_ = straxen.URLConfig(
        default=int(1e7),
        type=int,
        help="Search for ambience in this time window [ns]",
    )

    ambience_peak_selection = straxen.URLConfig(
        default="type==2",
        type=(str, None),
        help="Selection string for ambience peaks",
    )

    ambience_two_sides = straxen.URLConfig(
        default=False,
        type=bool,
        help="Use two sided ambience",
    )

    ambience_max_peak = straxen.URLConfig(
        default=1e4,
        type=(int, float),
        help="Exclude larger peaks than this from ambience peak [PE]",
    )

    ambience_min_area_fraction = straxen.URLConfig(
        default=0.5,
        help="Exclude smaller peaks than this fraction from ambience peak [PE]",
    )

    ambience_exponents = straxen.URLConfig(
        default=(1.0, -1.0, -1.0, 1.0),
        type=(list, tuple),
        help="The exponent of (primary peak, delta t, delta r) when calculating ambience",
    )

    ambience_exclude_scintillation = straxen.URLConfig(
        default=True,
        type=bool,
        help="Exclude S1 in (time, space) score",
    )

    ambience_exclude_sparse = straxen.URLConfig(
        default=True,
        type=bool,
        help="Exclude peaks sparse in position",
    )

    # TODO: if the drift velocity is not strictly needed, it should be removed
    electron_drift_velocity = straxen.URLConfig(
        default="xedocs://electron_drift_velocities?attr=value&run_id=plugin.run_id&version=ONLINE",
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z,
        type=(int, float),
        help="Total length of the TPC from the bottom of gate to the top of cathode wires [cm]",
    )

    def get_window_size(self):
        return (10 * self.ambience_time_window_, 0)

    def infer_dtype(self):
        dtype = [
            (
                ("Strength of ambient peaks only in time", "ambience_1d_score"),
                np.float32,
            ),
            (
                ("Strength of ambient peaks in (time, space)", "ambience_2d_score"),
                np.float32,
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def setup(self):
        self.to_pe = self.gain_model

        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)

    def compute(self, peaks, lone_hits):
        argsort = strax.stable_argsort(peaks["center_time"])
        _peaks = peaks[argsort].copy()
        result = np.zeros(len(peaks), self.dtype)
        _quick_assign(argsort, result, self.compute_ambience(peaks, lone_hits, _peaks))
        return result

    def compute_ambience(self, peaks, lone_hits, current_peak):
        roi = np.zeros(len(current_peak), dtype=strax.time_fields)
        roi["time"] = current_peak["center_time"] - self.ambience_time_window_
        if self.ambience_two_sides:
            roi["endtime"] = current_peak["center_time"] + self.ambience_time_window_
        else:
            roi["endtime"] = current_peak["center_time"]

        result = np.zeros(len(current_peak), dtype=self.dtype)
        strax.set_nan_defaults(result)
        result["time"] = current_peak["time"]
        result["endtime"] = current_peak["endtime"]

        primary = parse_selection(peaks, self.ambience_peak_selection)
        if self.ambience_exclude_sparse:
            primary &= ~np.isin(
                peaks["type"],
                [FAR_XYPOS_S2_TYPE, WIDE_XYPOS_S2_TYPE],
            )
        primary &= peaks["area"] < self.ambience_max_peak

        result["ambience_1d_score"] = self.peaks_ambience(
            current_peak,
            peaks[primary],
            strax.touching_windows(peaks[primary], roi),
            self.ambience_exponents,
            min_area_fraction=self.ambience_min_area_fraction,
            divide_r=False,
        )

        area_lh = lone_hits["area"] * self.to_pe[lone_hits["channel"]]
        primary_lh = area_lh < self.ambience_max_peak
        dtype = np.dtype(
            [
                ("time", np.int64),
                ("endtime", np.int64),
                ("center_time", np.int64),
                ("area", np.float32),
                ("x", np.float32),
                ("y", np.float32),
            ]
        )
        lh = np.zeros(primary_lh.sum(), dtype=dtype)
        lh["time"] = lone_hits["time"][primary_lh]
        lh["endtime"] = strax.endtime(lone_hits)[primary_lh]
        lh["center_time"] = lh["time"]
        lh["area"] = area_lh[primary_lh]
        ambience_1d_score_lh = self.peaks_ambience(
            current_peak,
            lh,
            strax.touching_windows(lh, roi),
            self.ambience_exponents,
            min_area_fraction=self.ambience_min_area_fraction,
            divide_r=False,
        )
        result["ambience_1d_score"] += ambience_1d_score_lh

        if self.ambience_exclude_scintillation:
            primary &= peaks["type"] == 2
        result["ambience_2d_score"] = self.peaks_ambience(
            current_peak,
            peaks[primary],
            strax.touching_windows(peaks[primary], roi),
            self.ambience_exponents,
            min_area_fraction=0,
            divide_r=True,
        )
        return result

    @staticmethod
    @numba.njit
    def peaks_ambience(
        peaks,
        pre_peaks,
        touching_windows,
        exponents,
        min_area_fraction=0.0,
        divide_r=False,
    ):
        sum_array = np.zeros(len(peaks), np.float32)
        for p_i, suspicious_peak in enumerate(peaks):
            indices = touching_windows[p_i]
            for idx in range(indices[0], indices[1]):
                creating_peak = pre_peaks[idx]
                dt = suspicious_peak["center_time"] - creating_peak["center_time"]
                if (dt == 0) or (creating_peak["area"] <= 0) or (suspicious_peak["area"] <= 0):
                    continue
                if creating_peak["area"] < min_area_fraction * suspicious_peak["area"]:
                    continue
                score = creating_peak["area"] ** exponents[0] * np.abs(dt) ** exponents[1]
                # score = creating_peak["area"] > min_area_fraction * suspicious_peak["area"]
                if divide_r:
                    r = distance_in_xy(suspicious_peak, creating_peak)
                    if np.isnan(r):
                        continue
                    score *= r ** exponents[2]
                score /= suspicious_peak["area"] ** exponents[3]
                sum_array[p_i] += score
        return sum_array
