import numpy as np
import numba
import strax
import straxen
from typing import List
from .peak_ambience import _quick_assign

export, __all__ = strax.exporter()


@export
class PeakLoneHitsShadow(strax.OverlapWindowPlugin):
    """
    Plugin to compute the lone hits shadow for each peak in the peaks array
    ref: xenon:xenonnt_sr0:further_ac_suppression_hits
    """

    __version__ = "0.0.2"

    depends_on = "peak_basics"
    provides = "peak_lone_hits_shadow"
    save_when = strax.SaveWhen.EXPLICIT

    lone_hits_shadow_look_back_window = straxen.URLConfig(
        default=int(1e9),
        type=int,
        track=True,
        help="Look back window for the lone hits shadow function",
    )

    lone_hits_shadow_pre_peak_threshold = straxen.URLConfig(
        default=20000.0,
        type=float,
        track=True,
        help="Threshold area to check lone hit shadow for the prepeak",
    )

    lone_hits_shadow_model_bounds = straxen.URLConfig(
        default=[4.50, 200.0],
        type=list,
        track=True,
        help="Lone hit shadow modeling time bounds [us]",
    )

    lone_hits_total_density_params = straxen.URLConfig(
        default=dict(a=0.0015, b=500.0),
        type=dict,
        track=True,
        help="Absolute number of lone hits as the function of the prepeak area",
    )

    lone_hits_time_profile_tau = straxen.URLConfig(
        default=dict(
            A=46.583791417437006,
            tau1=-0.020436220210423274,
            B=-18.62021103185991,
            tau2=-0.09723764851542319,
            C=-31.096252249643133,
        ),
        type=dict,
        track=True,
        help="Decay tau of the lone hits shadow function as the function of the prepeak area",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (("Modeled local lone_hits density", "peak_lone_hits_shadow"), np.float32)
        ]
        return dtype

    def get_window_size(self):
        return 10 * self.lone_hits_shadow_look_back_window

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def parameterization_tau(pre_peak_area, A, tau1, B, tau2, C):
        """
        Parameterization of the area dependence of the A and tau in the lone hits shadow function
        :param pre_peak_area:
        :param A:
        :param tau1:
        :param B:
        :param tau2:
        :param C:
        :return:
        """
        return A * pre_peak_area**tau1 + B * pre_peak_area**tau2 + C

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def lone_hits_shadow_function(pre_peak_area, time_diff, a, b, tau, bounds):
        return (
            (a * pre_peak_area + b)
            * (tau + 1)
            / (bounds[1] ** (tau + 1) - bounds[0] ** (tau + 1))
            * time_diff**tau
        )

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def peak_lone_hits_shadow(
        peaks,
        pre_peaks,
        parameterization_tau,
        lone_hits_shadow_function,
        touching_windows,
        a,
        b,
        A,
        tau1,
        B,
        tau2,
        C,
        bounds,
        _result,
    ):
        for i, peak in enumerate(peaks):
            indices = touching_windows[i]
            for j in range(indices[0], indices[1]):
                pre_peak = pre_peaks[j]
                time_diff = (peak["time"] - pre_peak["time"]) / 1e6  # ms
                if time_diff < 0:
                    continue
                tau = parameterization_tau(pre_peak["area"], A, tau1, B, tau2, C)
                lone_hits_shadow = lone_hits_shadow_function(
                    pre_peak["area"],
                    time_diff,
                    a,
                    b,
                    tau,
                    bounds,
                )
                if lone_hits_shadow > _result["peak_lone_hits_shadow"][i]:
                    _result["peak_lone_hits_shadow"][i] = lone_hits_shadow

    def searching_window(self, _peaks, look_back):
        """
        Create a searching window for each peak in the peaks array
        :param peaks:
        :param look_back
        :return:
        """
        search_window = np.zeros(len(_peaks), dtype=strax.time_fields)
        search_window["time"] = _peaks["time"] - look_back
        search_window["endtime"] = _peaks["time"]
        return search_window

    def compute(self, peaks):
        result = np.zeros_like(peaks, dtype=self.infer_dtype())

        argsort = np.argsort(peaks["time"], kind="mergesort")
        _peaks = np.sort(peaks, order="time")
        _result = np.zeros_like(_peaks, dtype=self.infer_dtype())
        _result["time"] = _peaks["time"]
        _result["endtime"] = strax.endtime(_peaks)

        pre_peaks = _peaks[_peaks["area"] >= self.lone_hits_shadow_pre_peak_threshold]

        touching_windows = strax.touching_windows(
            pre_peaks,
            self.searching_window(_peaks, self.lone_hits_shadow_look_back_window),
        )

        self.peak_lone_hits_shadow(
            peaks=_peaks,
            pre_peaks=pre_peaks,
            parameterization_tau=self.parameterization_tau,
            lone_hits_shadow_function=self.lone_hits_shadow_function,
            touching_windows=touching_windows,
            a=np.float64(self.lone_hits_total_density_params["a"]),
            b=np.float64(self.lone_hits_total_density_params["b"]),
            A=np.float64(self.lone_hits_time_profile_tau["A"]),
            tau1=np.float64(self.lone_hits_time_profile_tau["tau1"]),
            B=np.float64(self.lone_hits_time_profile_tau["B"]),
            tau2=np.float64(self.lone_hits_time_profile_tau["tau2"]),
            C=np.float64(self.lone_hits_time_profile_tau["C"]),
            bounds=self.lone_hits_shadow_model_bounds,
            _result=_result,
        ),

        _quick_assign(argsort, result, _result)

        return result
