import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@export
class EventTopBottomParams(strax.Plugin):
    """Pluging that computes timing characteristics of top and bottom waveforms based on waveforms
    stored at event level for main/alt S1/S2."""

    depends_on = ("event_info", "event_waveform")
    provides = "event_top_bottom_params"
    __version__ = "0.0.0"

    def infer_dtype(self):
        # Populating data type information
        infoline = {
            "s1": "main S1",
            "s2": "main S2",
            "alt_s1": "alternative S1",
            "alt_s2": "alternative S2",
        }
        ev_info_fields = self.deps["event_info"].dtype.fields
        dtype = []
        # populating APC and waveform samples
        self.ptypes = ["s1", "s2", "alt_s1", "alt_s2"]
        self.arrs = ["top", "bot"]
        for type_ in self.ptypes:
            for arr_ in self.arrs:
                dtype += [
                    (
                        (
                            f"Central time for {infoline[type_]} for {arr_} PMTs [ ns ]",
                            f"{type_}_center_time_{arr_}",
                        ),
                        ev_info_fields[f"{type_}_center_time"][0],
                    )
                ]
                dtype += [
                    (
                        (
                            (
                                f"Time between 10% and 50% area quantiles for {infoline[type_]} for"
                                f" {arr_} PMTs [ns]"
                            ),
                            f"{type_}_rise_time_{arr_}",
                        ),
                        ev_info_fields[f"{type_}_rise_time"][0],
                    )
                ]
                dtype += [
                    (
                        (
                            (
                                f"Width (in ns) of the central 50% area of the peak for {arr_} PMTs"
                                f" of {infoline[type_]}"
                            ),
                            f"{type_}_range_50p_area_{arr_}",
                        ),
                        ev_info_fields[f"{type_}_range_50p_area"][0],
                    )
                ]
                dtype += [
                    (
                        (
                            (
                                f"Width (in ns) of the central 90% area of the peak for {arr_} PMTs"
                                f" of {infoline[type_]}"
                            ),
                            f"{type_}_range_90p_area_{arr_}",
                        ),
                        ev_info_fields[f"{type_}_range_90p_area"][0],
                    )
                ]
            dtype += [
                (
                    (
                        (
                            "Difference between center times of top and bottom arrays for"
                            f" {infoline[type_]} [ ns ]"
                        ),
                        f"{type_}_center_time_diff_top_bot",
                    ),
                    ev_info_fields[f"{type_}_center_time"][0],
                )
            ]
        dtype += strax.time_fields
        return dtype

    def compute(self, events):
        result = np.zeros(events.shape, dtype=self.dtype)
        result["time"], result["endtime"] = events["time"], strax.endtime(events)
        peak_dtype = strax.peak_dtype(n_channels=straxen.n_tpc_pmts, store_data_top=False)
        for type_ in self.ptypes:
            for arr_ in self.arrs:
                # in order to reuse the same definitions as in other parts, we create "fake peaks"
                # based only on data from corresponding array
                fpeaks_ = np.zeros(events.shape[0], dtype=peak_dtype)
                if arr_ == "top":
                    fpeaks_["data"] = events[f"{type_}_data_top"]
                    fpeaks_["area"] = events[f"{type_}_area"] * events[f"{type_}_area_fraction_top"]
                elif arr_ == "bot":
                    fpeaks_["data"] = events[f"{type_}_data"] - events[f"{type_}_data_top"]
                    fpeaks_["area"] = events[f"{type_}_area"] * (
                        1.0 - events[f"{type_}_area_fraction_top"]
                    )
                elif arr_ == "tot":
                    # This one is ony
                    fpeaks_["data"] = events[f"{type_}_data"]
                    fpeaks_["area"] = events[f"{type_}_area"]
                else:
                    raise RuntimeError(f"Received unknown array type : " + arr_)
                fpeaks_["length"] = events[f"{type_}_length"]
                fpeaks_["dt"] = events[f"{type_}_dt"]
                # computing central times
                # note that here we ignore 1/2 sample length to be consistent with other definitions
                with np.errstate(divide="ignore", invalid="ignore"):
                    recalc_ctime = np.sum(
                        fpeaks_["data"] * (np.arange(0, fpeaks_["data"].shape[1])), axis=1
                    )
                    recalc_ctime /= fpeaks_["area"]
                    recalc_ctime *= fpeaks_["dt"]
                    recalc_ctime[~(fpeaks_["area"] > 0)] = 0.0
                # setting central times in the same way as inside peak processing
                mask = fpeaks_["area"] > 0
                result[f"{type_}_center_time_{arr_}"] = events[f"{type_}_time"]
                result[f"{type_}_center_time_{arr_}"][mask] += recalc_ctime[mask].astype(int)
                # computing widths ##
                # zero or undefined area peaks should have nans
                strax.compute_properties(fpeaks_)
                result[f"{type_}_rise_time_{arr_}"][:] = np.nan
                result[f"{type_}_rise_time_{arr_}"][mask] = -fpeaks_["area_decile_from_midpoint"][
                    mask
                ][:, 1]
                result[f"{type_}_range_50p_area_{arr_}"][:] = np.nan
                result[f"{type_}_range_50p_area_{arr_}"][mask] = fpeaks_["width"][mask][:, 5]
                result[f"{type_}_range_90p_area_{arr_}"][:] = np.nan
                result[f"{type_}_range_90p_area_{arr_}"][mask] = fpeaks_["width"][mask][:, 9]
            # Difference between center times of top and bottom arrays
            result[f"{type_}_center_time_diff_top_bot"] = (
                result[f"{type_}_center_time_top"] - result[f"{type_}_center_time_bot"]
            )
        return result
