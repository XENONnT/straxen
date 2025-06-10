import numpy as np
import straxen
import strax

export, __all__ = strax.exporter()

MAX_NUMBER_OF_S1_PEAKS_PER_EVENT = 2
MAX_NUMBER_OF_S2_PEAKS_PER_EVENT = 5
PEAK_WAVEFORM_LENGTH = 200 
HIT_PATTERN_LENGTH = 494

@export
class MultiPeakMSData(strax.Plugin):
    """Plugin that extracts information of multiple peaks in an event and multi scatter features"""

    __version__ = "0.1.0"

    depends_on = (
                "peaks",
                "event_info",
                "peak_basics", 
                "peak_corrections",
                "peak_per_event", 
                "peak_positions", 
                 )

    provides = "multi_peak_ms_naive_data"
    data_kind = "events"
    save_when = strax.SaveWhen.TARGET
    
    electron_drift_velocity = straxen.URLConfig(
        default="cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    electron_drift_time_gate = straxen.URLConfig(
        default="cmt://electron_drift_time_gate?version=ONLINE&run_id=plugin.run_id",
        help="Electron drift time from the gate in ns",
        cache=True,
    )

    fdc_map = straxen.URLConfig(
        default="legacy-fdc://xenon1t_sr0_sr1?run_id=plugin.run_id",
        infer_type=False,
        help="3D field distortion correction map path",
    )

    z_bias_map = straxen.URLConfig(
        default="legacy-z_bias://0",
        infer_type=False,
        help="Map of Z bias due to non uniform drift velocity/field",
    )

    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z,
        infer_type=False,
        help="Total length of the TPC from the bottom of gate to the top of cathode wires [cm]",
    )

    ms_window_fac = straxen.URLConfig(
        default=1.01,
        type=(int, float),
        help="Max drift time window to look for peaks in multiple scatter events",
    )
        
    dtype = [  
    (("time difference of S1 peaks to event start time", "s1_delta_time_i"), np.int64, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
    (("time difference of S2 peaks to event start time", "s2_delta_time_i"), np.int64, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("sample width in ns of the S1 waveform", "s1_peak_dt_i"), np.int32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
    (("sample width in ns of the S2 waveform", "s2_peak_dt_i"), np.int32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("x position of S2 i", "s2_x_position_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("y position of S2 i", "s2_y_position_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("z position of S2 i", "s2_z_position_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("Corrected x position of S2 i", "s2_x_position_corr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("Corrected y position of S2 i", "s2_y_position_corr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("Corrected z position of S2 i", "s2_z_position_corr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),   
    (("Corrected r position of S2 i", "s2_r_position_corr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("Area fraction top of S1 i", "s1_aft_i"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
    (("Area fraction top of S2 i", "s2_aft_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("Sum Waveform of S1 peaks", "s1_waveform_i"), np.float32, (MAX_NUMBER_OF_S1_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH)),
    (("Sum Waveform of S2 peaks", "s2_waveform_i"), np.float32, (MAX_NUMBER_OF_S2_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH)),
    (("PMT Hitpattern of S2 peaks", "s2_area_per_channel_i"), np.float32, (MAX_NUMBER_OF_S2_PEAKS_PER_EVENT, HIT_PATTERN_LENGTH)),
    (("Area of S2 i", "s2_area_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("Area of S1 i", "s1_area_i"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
    (("Corrected area of S2 i", "cs2_area_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("Corrected area fraction top of S2 i", "cs2_aft_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    (("Corrected area fraction top w/o time correction of S2 i", "cs2_wo_timecorr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
    *strax.time_fields,
    ]

    def setup(self):
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)
        self.coordinate_scales = [1.0, 1.0, -self.electron_drift_velocity]
        self.map = self.fdc_map

    def correct_positions(self, x, y, z):
        """
        This function, inspired by the event_positions method, 
        corrects the event positions by applying Z-bias and FDC corrections.

        Parameters:
        -----------
        x, y, z : array_like

        Returns:
        --------
        x_cor, y_cor, z_cor, r_cor : array_like
        """        
        orig_pos = np.vstack([x, y, z]).T
        r_obs = np.linalg.norm(orig_pos[:, :2], axis=1)
        z += self.electron_drift_velocity * self.electron_drift_time_gate
            
        # apply Z bias correction
        z_dv_delta = self.z_bias_map(np.array([r_obs, z]).T, map_name="z_bias_map")
        corr_pos = np.vstack([x, y, z - z_dv_delta]).T  # (N, 3)
        # apply FDC correction
        delta_r = self.map(corr_pos)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_cor = r_obs + delta_r
            scale = np.divide(r_cor, r_obs, out=np.zeros_like(r_cor), where=r_obs != 0)
        
        x_cor = x * scale
        y_cor = y * scale
        z_cor = z - z_dv_delta
        r_cor = r_cor

        return x_cor, y_cor, z_cor, r_cor
    
    def compute(self, peaks, events):

        peaks = peaks[peaks["type"] != 0]
        split_peaks = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), dtype=self.dtype)

        for i, (event, peaks_in_event) in enumerate(zip(events, split_peaks)):

            if len(peaks_in_event) == 0:
                continue

            s1_peaks = peaks_in_event[peaks_in_event["type"] == 1]
            s2_peaks = peaks_in_event[peaks_in_event["type"] == 2]

            s1_peaks = s1_peaks[np.argsort(s1_peaks["area"])[::-1]]
            s2_peaks = s2_peaks[np.argsort(s2_peaks["area"])[::-1]]

            n_s1_peaks_in_event = len(s1_peaks)
            n_s2_peaks_in_event = len(s2_peaks)

            result[i]["s1_delta_time_i"][:n_s1_peaks_in_event] = s1_peaks["time"][:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT] - peaks_in_event["time"][0]
            result[i]["s2_delta_time_i"][:n_s2_peaks_in_event] = s2_peaks["time"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT] - peaks_in_event["time"][0]

            result[i]["s1_peak_dt_i"][:n_s1_peaks_in_event] = s1_peaks["dt"][:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT]
            result[i]["s2_peak_dt_i"][:n_s2_peaks_in_event] = s2_peaks["dt"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]

            x_obs = s2_peaks["x"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]
            y_obs = s2_peaks["y"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]
            z_obs = s2_peaks["z_obs_ms"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]
            
            result[i]["s2_x_position_i"][:n_s2_peaks_in_event] = x_obs
            result[i]["s2_y_position_i"][:n_s2_peaks_in_event] = y_obs
            result[i]["s2_z_position_i"][:n_s2_peaks_in_event] = z_obs

            (
                result[i]["s2_x_position_corr_i"][:n_s2_peaks_in_event],
                result[i]["s2_y_position_corr_i"][:n_s2_peaks_in_event],
                result[i]["s2_z_position_corr_i"][:n_s2_peaks_in_event],
                result[i]["s2_r_position_corr_i"][:n_s2_peaks_in_event],
            ) = self.correct_positions(x_obs, y_obs, z_obs)
        
            result[i]["s1_aft_i"][:n_s1_peaks_in_event] = s1_peaks["area_fraction_top"][:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT]
            result[i]["s2_aft_i"][:n_s2_peaks_in_event] = s2_peaks["area_fraction_top"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]

            result[i]["s1_waveform_i"][:n_s1_peaks_in_event] = s1_peaks["data"][:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT]
            result[i]["s2_waveform_i"][:n_s2_peaks_in_event] = s2_peaks["data"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]

            result[i]["s2_area_per_channel_i"][:n_s2_peaks_in_event] = s2_peaks["area_per_channel"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]
            result[i]["s2_area_i"][:n_s2_peaks_in_event] = s2_peaks["area"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]
            result[i]["s1_area_i"][:n_s1_peaks_in_event] = s1_peaks["area"][:MAX_NUMBER_OF_S1_PEAKS_PER_EVENT]
            result[i]["cs2_area_i"][:n_s2_peaks_in_event] =  s2_peaks["cs2"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT] 
            result[i]['cs2_aft_i'][:n_s2_peaks_in_event] =  s2_peaks["cs2_area_fraction_top"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT] 
            result[i]["cs2_wo_timecorr_i"][:n_s2_peaks_in_event] =  s2_peaks["cs2_wo_timecorr"][:MAX_NUMBER_OF_S2_PEAKS_PER_EVENT]
            
        return result

        