import numpy as np
import strax
import straxen
from straxen.plugins.defaults import NON_TRIGGERABLE_S1_TYPE, NON_TRIGGERABLE_S2_TYPE


class TriggerablePeakBasics(strax.Plugin):
    __version__ = "0.0.0"
    depends_on = ("peak_basics", "peak_shadow", "peak_se_score")
    provides = "triggerable_peak_basics"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        return self.deps["peak_basics"].dtype_for("peak_basics")

    sr = straxen.URLConfig(
        default="science_run://plugin.run_id?&phase=False",
        help="Science run to be used for the cut. It can affect the parameters of the cut.",
    )

    run_mode = straxen.URLConfig(
        default="run_mode://plugin.run_id",
        help="Run mode to be used for the cut. It can affect the parameters of the cut.",
    )

    # TimeVeto
    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z,
        type=(int, float),
        help="Total length of the TPC from the bottom of gate to the top of cathode wires [cm]",
    )

    shadow_threshold = straxen.URLConfig(
        default={"s1_time_shadow": 1e3, "s2_time_shadow": 1e4, "s2_position_shadow": 1e4},
        type=dict,
        track=True,
        help="Only take S1/S2s larger than this into account when calculating Shadow [PE]",
    )

    electron_drift_velocity = straxen.URLConfig(
        default="xedocs://electron_drift_velocities?attr=value&run_id=plugin.run_id&version=ONLINE",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    n_drift_time = straxen.URLConfig(
        default={"sr0": 1, "sr1": 2}, help="Number of drift time to veto"
    )

    # TimeShadow
    time_shadow_cut_values_mode = straxen.URLConfig(
        default={
            "bkg": [0.01010, 0.03822],  # Used by bkg and ted runs
            "ar37": [0.01010, 0.03822],
            "radon": [0.08303, 0.40418],
            "radon_hev": [0.33115, 0.61039],
            "ambe": [0.03078, 0.18748],
            "kr83m": [0.01479, 0.07908],
            "ybe": [0.03189, 0.09658],
            "rn222": [0.01178, 0.02782],
        },
        type=dict,
        track=True,
        help="cut values for 2 & 3 hits S1 in different run mode",
    )

    # PositionShadow
    position_shadow_s2_area_limit = straxen.URLConfig(
        default=5e4,
        type=float,
        track=True,
        help="Max S2 area to implement this cut, LowER analysis upper limit [PE]",
    )

    position_shadow_ellipse_parameters_mode = straxen.URLConfig(
        default={
            "bkg": [-0.969, -2.485, 0.564, 0.766, 1.798],  # Used by bkg and ted runs
            "ar37": [-0.969, -2.485, 0.564, 0.766, 1.798],
            "radon": [-0.895, -2.226, 0.713, 0.818, 1.370],
            "radon_hev": [-0.873, -2.192, 0.624, 0.795, 1.504],
            "ambe": [-0.911, -1.947, 0.821, 1.047, 0.734],
            "kr83m": [-0.862, -2.531, 0.652, 0.779, 1.922],
            "ybe": [-0.961, -2.344, 0.796, 0.885, 1.534],
            "rn222": [-0.940, -2.537, 0.719, 0.829, 1.612],
        },
        type=dict,
        track=True,
        help="cut parameters for the ellipse curve in different run mode",
    )

    position_shadow_straight_parameters_mode = straxen.URLConfig(
        default={
            "bkg": [-0.947, -4.231],  # Used by bkg and ted runs
            "ar37": [-0.947, -4.231],
            "radon": [-0.975, -3.388],
            "radon_hev": [-0.868, -3.110],
            "ambe": [-0.430, -1.993],
            "kr83m": [-1.022, -4.195],
            "ybe": [-0.801, -2.670],
            "rn222": [-0.929, -4.009],
        },
        type=dict,
        track=True,
        help="cut parameters for the straight line in different run mode",
    )

    # HotspotVeto
    ambience_hotspot_veto_threshold = straxen.URLConfig(
        default={"s1": 150, "s2": 1e12},
        type=dict,
        track=True,
        help="Do not apply the cut if S1/2 is larger than this",
    )

    hotspot_veto_threshold = straxen.URLConfig(
        default=0.1,
        type=(int, float),
        help="Upper boundary of SE score",
    )

    def setup(self):
        # Force using background mode parameters
        # self.run_mode = 'bkg'
        # TimeVeto
        self.veto_window = (
            self.max_drift_length / self.electron_drift_velocity * self.n_drift_time[self.sr]
        )
        # TimeShadow
        self.time_shadow = self.time_shadow_cut_values_mode[self.run_mode]
        # PositionShadow
        self.s2_area_limit = self.position_shadow_s2_area_limit
        self.ellipse_parameters = self.position_shadow_ellipse_parameters_mode[self.run_mode]
        self.straight_parameters = self.position_shadow_straight_parameters_mode[self.run_mode]
        # HotspotVeto
        if self.run_mode != "bkg":
            self.threshold = np.inf
        else:
            self.threshold = self.hotspot_veto_threshold

    @staticmethod
    @np.errstate(invalid="ignore")
    def prob_cut_line(log10_pdf, x0, y0, a, b, rlimit, pstraight):
        top = 0
        # The top line in 2D histogram
        topline = np.full(len(log10_pdf), top)

        # Draw a half ellipse
        halfellipse = np.where(
            np.abs(log10_pdf - x0) < a * rlimit,
            y0 - np.sqrt(b**2 * (rlimit**2 - (log10_pdf - x0) ** 2 / a**2)),
            top,
        )

        # A horizontal tangent line at the bottom of ellipse
        botline = np.where(log10_pdf > x0, y0 - b * rlimit, top)

        # A straight line with slope
        straight = np.polyval(pstraight, log10_pdf)

        # The lowest of 4 lines
        line = np.min(np.vstack([topline, halfellipse, botline, straight]), axis=0)
        return line

    @np.errstate(divide="ignore")
    def compute(self, peaks):
        result = np.zeros(len(peaks), dtype=self.dtype)
        strax.set_nan_defaults(result)
        strax.copy_to_buffer(peaks, result, "_copy_peak_basics_information")

        # Only do something for the background runs
        if self.run_mode != "bkg":
            return result

        # TimeVeto
        # TODO: do not apply S1 shadow on S2
        # or at least do not search for S1 within max drift time
        time_veto = np.full(len(peaks), True)
        for casting_peak in ["_s1", "_s2"]:
            time_veto &= (peaks[f"nearest_dt{casting_peak}"] > self.veto_window) | (
                peaks[f"nearest_dt{casting_peak}"] < 0
            )
        time_veto |= (peaks["type"] == 1) & (
            peaks["area"] > self.shadow_threshold["s1_time_shadow"]
        )
        # this line is added additional to cutax
        # in cutax we are sure that when S1 is large enough, the event will bypass the veto
        # here we have to bypass veto on S2 by hand
        time_veto |= (peaks["type"] == 2) & (
            peaks["area"] > self.shadow_threshold["s2_time_shadow"]
        )

        # TimeShadow
        # here only 3-hits shadow cut is applied
        # TODO: loosen the threshold
        time_shadow_3hits = peaks["shadow_s2_time_shadow"] > self.time_shadow[1]
        time_shadow = ~time_shadow_3hits
        # Do not apply on high energy peaks, this is not the same to cutax
        # in cutax we have cut list so in som analysis we do not use shadow, like high energy
        # but here the event building will be applied to all analysis
        time_shadow |= (peaks["type"] == 1) & (
            peaks["area"] > self.shadow_threshold["s1_time_shadow"]
        )
        time_shadow |= (peaks["type"] == 2) & (
            peaks["area"] > self.shadow_threshold["s2_time_shadow"]
        )

        # PositionShadow
        log10_time_shadow = np.log10(
            peaks["shadow_s2_position_shadow"] / peaks["pdf_s2_position_shadow"]
        )
        log10_pdf = np.log10(peaks["pdf_s2_position_shadow"])
        position_shadow = ~(
            log10_time_shadow
            > self.prob_cut_line(
                log10_pdf,
                *self.ellipse_parameters,
                self.straight_parameters,
            )
        )
        position_shadow |= peaks["area"] > self.s2_area_limit
        position_shadow |= peaks["type"] != 2
        # Do not apply on high energy peaks
        position_shadow |= (peaks["type"] == 2) & (
            peaks["area"] > self.shadow_threshold["s2_time_shadow"]
        )

        # HotspotVeto
        hotspot_veto = peaks["se_score"] > self.threshold
        hotspot_veto = ~hotspot_veto
        # Do not apply on any S1 for now
        # hotspot_veto |= (peaks["type"] == 1) & (
        #     peaks["area"] > self.ambience_hotspot_veto_threshold["s1"]
        # )
        hotspot_veto |= (peaks["type"] == 2) & (
            peaks["area"] > self.ambience_hotspot_veto_threshold["s2"]
        )
        hotspot_veto |= (peaks["type"] == 2) & (
            peaks["area"] > self.shadow_threshold["s2_time_shadow"]
        )
        hotspot_veto |= peaks["type"] != 2

        mask = time_veto & time_shadow & position_shadow & hotspot_veto

        result["type"][~mask & (result["type"] == 1)] = NON_TRIGGERABLE_S1_TYPE
        result["type"][~mask & (result["type"] == 2)] = NON_TRIGGERABLE_S2_TYPE

        return result
