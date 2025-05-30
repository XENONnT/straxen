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
                "event_info",
                "events", 
                "corrected_areas",
                "event_basics",
                "event_positions",
                "peak_positions", 
                "peak_basics", 
                "peak_corrections",
                "peaks",
                "peak_per_event", 
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
        (("Sum of S1 areas in event", "s1_sum"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected S1 area based on average position of S2s in event", "cs1_multi"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected S1 area based on average position of S2s in event before time-dep LY correction", "cs1_multi_wo_timecorr"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Sum of S2 areas in event", "s2_sum"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Sum of corrected S2 areas in event", "cs2_sum"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected area of S2 i", "cs2_area_i"),  np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected AFT per S2 i","cs2_aft_i"),  np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected area wo time correctionper S2 i","cs2_wo_timecorr_i"),  np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Sum of corrected S2 areas in event S2 before elife correction", "cs2_wo_timecorr_sum",), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Sum of corrected S2 areas in event before SEG/EE and elife corrections", "cs2_wo_elifecorr_sum",), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Average of S2 area fraction top in event", "cs2_area_fraction_top_avg"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Sum of the energy estimates in event", "ces_sum"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Sum of the charge estimates in event", "e_charge_sum"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Average x position of S2s in event", "x_avg"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Average y position of S2s in event", "y_avg"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Average observed z position of energy deposits in event", "z_obs_avg"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Number of S2s in event", "multiplicity"), np.int32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Main S2 width, 50% area [ns]", "s2_range_50p_area"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Main S1 width, 50% area [ns]", "s1_range_50p_area"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (("Main S1 width, 90% area [ns]", "s1_range_90p_area"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Main S1 area fraction top", "s1_area_fraction_top"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (("Main interaction r-position, field-distortion corrected [cm]", "r"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Main interaction z-position, field-distortion corrected [cm]", "z"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Alternate S1 area, uncorrected [PE]", "alt_s1_area"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (("Alternate S2 area, uncorrected [PE]", "alt_s2_area"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Main S1 center time [ns]", "s1_center_time"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (("Main S1 time [ns]", "s1_time"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (("Drift time between main S1 and S2 [ns]", "drift_time"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Main S1 area [PE]", "s1_area"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (("Main S2 area [PE]", "s2_area"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected area of alternate S2 [PE]", "alt_cs2"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Fraction of area seen by the top PMT array for corrected alternate S2", "alt_cs2_area_fraction_top_wo_timecorr"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Fraction of area seen by the top PMT array for corrected main S2", "cs2_area_fraction_top_wo_timecorr"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected area of alternate S2 (before SEG/EE + photon ionization, after S2 xy + elife) [PE]", "alt_cs2_wo_timecorr"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("time difference of S1 peaks to event start time", "s1_delta_time_i"), np.int64, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT,),
        (("time difference of S2 peaks to event start time", "s2_delta_time_i"), np.int64, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT,),
        (("sample width in ns of the S1 waveform", "s1_peak_dt_i"), np.int32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT,),
        (("sample width in ns of the S2 waveform", "s2_peak_dt_i"), np.int32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT,),
        (("x position of S2 i", "s2_x_position_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("y position of S2 i", "s2_y_position_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("z position of S2 i", "s2_z_position_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected x position of S2 i", "s2_x_position_corr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected y position of S2 i", "s2_y_position_corr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected z position of S2 i", "s2_z_position_corr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("Corrected r position of S2 i", "s2_r_position_corr_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("area fraction top of S2 i", "s2_aft_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("area fraction top of S1 i", "s1_aft_i"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (("area of S2 i", "s2_area_i"), np.float32, MAX_NUMBER_OF_S2_PEAKS_PER_EVENT),
        (("area of S1 i", "s1_area_i"), np.float32, MAX_NUMBER_OF_S1_PEAKS_PER_EVENT),
        (("Sum Waveform of S1 peaks", "s1_waveform_i"), np.float32, (MAX_NUMBER_OF_S1_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH),),
        (("Sum Waveform of S2 peaks", "s2_waveform_i"), np.float32, (MAX_NUMBER_OF_S2_PEAKS_PER_EVENT, PEAK_WAVEFORM_LENGTH),),
        (("PMT Hitpattern of S2 Peaks", "s2_area_per_channel_i"), np.float32, (MAX_NUMBER_OF_S2_PEAKS_PER_EVENT, HIT_PATTERN_LENGTH),),
        
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

            n_peaks_in_event = len(peaks_in_event)

            result[i]["s1_area"][:n_peaks_in_event] = event["s1_area"]
            result[i]["s2_area"][:n_peaks_in_event] = event["s2_area"]
            result[i]["r"][:n_peaks_in_event] = event["r"]
            result[i]["z"][:n_peaks_in_event] = event["z"]
            result[i]["s1_range_90p_area"][:n_peaks_in_event] = event["s1_range_90p_area"]
            result[i]["s1_range_50p_area"][:n_peaks_in_event] = event["s1_range_50p_area"]
            result[i]["s2_range_50p_area"][:n_peaks_in_event] = event["s2_range_50p_area"]
            result[i]["s1_area_fraction_top"][:n_peaks_in_event] = event["s1_area_fraction_top"]
            result[i]["alt_s1_area"][:n_peaks_in_event] = event["alt_s1_area"]
            result[i]["alt_s2_area"][:n_peaks_in_event] = event["alt_s2_area"]
            result[i]["s1_center_time"][:n_peaks_in_event] = event["s1_center_time"]
            result[i]["s1_time"][:n_peaks_in_event] = event["s1_time"]
            result[i]["drift_time"][:n_peaks_in_event] = event["drift_time"]
            result[i]["s2_area"][:n_peaks_in_event] = event["s2_area"]
            result[i]["s1_area"][:n_peaks_in_event] = event["s1_area"]
            result[i]["alt_cs2"][:n_peaks_in_event] = event["alt_cs2"]
            result[i]["alt_cs2_wo_timecorr"][:n_peaks_in_event] = event["alt_cs2_wo_timecorr"]
            result[i]["alt_cs2_area_fraction_top_wo_timecorr"][:n_peaks_in_event] = event["alt_cs2_area_fraction_top_wo_timecorr"]
            result[i]["cs2_area_fraction_top_wo_timecorr"][:n_peaks_in_event] = event["cs2_area_fraction_top_wo_timecorr"]            

            cond = (peaks_in_event["type"] == 2) & (peaks_in_event["drift_time"] > 0)
            cond &= (peaks_in_event["drift_time"] < self.ms_window_fac * self.drift_time_max) & (peaks_in_event["cs2"] > 0)

            result[i]["s2_sum"][:n_peaks_in_event] = np.nansum(peaks_in_event[cond]["area"])
            result[i]["cs2_sum"][:n_peaks_in_event] = np.nansum(peaks_in_event[cond]["cs2"])
            
            result[i]["cs2_wo_timecorr_sum"][:n_peaks_in_event] = np.nansum(peaks_in_event[cond]["cs2_wo_timecorr"])
            result[i]["cs2_wo_elifecorr_sum"][:n_peaks_in_event] = np.nansum(peaks_in_event[cond]["cs2_wo_elifecorr"])
            result[i]["s1_sum"][:n_peaks_in_event] = np.nansum(peaks_in_event["area"]) 

            if np.sum(peaks_in_event[cond]["cs2"]) > 0:
                result[i]["cs1_multi_wo_timecorr"][:n_peaks_in_event] = event["s1_area"] * np.average(
                        peaks_in_event[cond]["s1_xyz_correction_factor"], weights=peaks_in_event[cond]["cs2"]
                    )
                result[i]["cs1_multi"][:n_peaks_in_event] = result[i]["cs1_multi_wo_timecorr"][:n_peaks_in_event] * np.average(
                        peaks_in_event[cond]["s1_rel_light_yield_correction_factor"], weights=peaks_in_event[cond]["cs2"]
                    )
                result[i]["x_avg"][:n_peaks_in_event] = np.average(peaks_in_event[cond]["x"], weights=peaks_in_event[cond]["cs2"])
                result[i]["y_avg"][:n_peaks_in_event] = np.average(peaks_in_event[cond]["y"], weights=peaks_in_event[cond]["cs2"])
                result[i]["z_obs_avg"][:n_peaks_in_event] = np.average(peaks_in_event[cond]["z_obs_ms"], weights=peaks_in_event[cond]["cs2"])
                result[i]["cs2_area_fraction_top_avg"][:n_peaks_in_event] = np.average(
                        peaks_in_event[cond]["cs2_area_fraction_top"], weights=peaks_in_event[cond]["cs2"]
                    )
                result[i]["multiplicity"][:n_peaks_in_event] = len(peaks_in_event[cond]["area"])

        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        return result