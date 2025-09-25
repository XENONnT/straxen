import strax
import numpy as np

export, __all__ = strax.exporter()


@export
class peak_tagging(strax.Plugin):
    """Gives tags to peaks, mainly in order to seperate physical S2s from not physical ones and
    e-train leakage.

    Look at note -
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:
    xenonnt:analysis:analysts_overview_page
    :roi_frankel:ms_v2

    """

    __version__ = "0.0.1"
    depends_on = ("event_basics", "peak_basics")
    provides = "peaks_tags"
    data_kind = "peaks"
    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        dtype = strax.time_fields + [
            ("drift_time", np.float32, "Drift time between main S1 and S2 in ns"),
            ("peak_tag", "U50", "The peak type, strings are relevant only for S2"),
        ]
        return dtype

    # fuction is used to devide between diffrent populations in the S2 area vs range_50p_area plane
    def linear_line(self, data, x1, x2, y1, y2):
        return (data - x1) * (y2 - y1) / (x2 - x1) + y1

    def compute(self, events, peaks):

        # plugin constants
        # first solid seperation limit points
        x_points_1spl = np.array([1.5e3, 2.5e3, 2.9e3, 3.3e3])
        y_points_1spl = np.array([5e1, 6e3, 8e3, 1e4])

        # second solid seperation limit point
        x_points_2spl = np.array(
            [3.3e3, 4e3, 5.5e3, 7.5e3, 1.03e4]
        )  # x points for the left-up line, second part of the S2 region seperation line
        y_points_2spl = np.array(
            [1e4, 1.15e4, 1.5e4, 2.4e4, 3.6e4]
        )  # y points for the left-up line, second part of the S2 region seperation line

        # Polynomial fits for the first and second solid seperation limit lines
        coefficients_1spl = np.polyfit(x_points_1spl, y_points_1spl, len(x_points_1spl) - 1)
        coefficients_2spl = np.polyfit(x_points_2spl, y_points_2spl, len(x_points_2spl) - 1)

        x_2_of_4spl = 1e9  # X-coordinate for high-area S2 linear separation (used in mask_S2_V3)
        y_2_of_4spl = 1.4e7  # Y-coordinate for high-area S2 linear separation (used in mask_S2_V3)
        v3 = 4e4  # Area threshold separating V2/V3 regions (used in mask_S2_V2, mask_S2_V3)
        stop_point_edge = (
            3.5e4  # Maximum area for edge region classification (used in mask_in_edge_2)
        )
        bottom_physical_horizontal_line_1 = 2.3e2  # Lower range_50p_area limit for
        # V0,V1,V2 regions (used in mask_S2_V0, mask_S2_V1, mask_S2_V2)
        bottom_physical_horizontal_line_2 = (
            2e2  # Lower range_50p_area limit for V3 region (used in mask_S2_V3)
        )
        y_mid_for_edge = 4e3  # Middle range_50p_area threshold for
        # edge regions (used in mask_in_edge_0, mask_in_edge_1)
        y_upper_for_edge = (
            5.6e3  # Upper range_50p_area threshold for edge region 2 (used in mask_in_edge_2)
        )
        y_2_of_3spl = (
            y_points_2spl[-1] + 2e3
        )  # Extended Y-value for linear separation in V2,V3 regions
        # (used in mask_S2_V2, mask_S2_V3, mask_in_edge_2)
        y_1 = np.polyval(
            coefficients_1spl, x_points_1spl[-1]  # Y-value at junction between polynomial curves
        )  # (used for offset calculation in mask_S2_V2, mask_in_edge_2)
        y_2 = np.polyval(
            coefficients_2spl, x_points_2spl[0]  # Y-value at start of second polynomial curve
        )  # (used for offset calculation in mask_S2_V2, mask_in_edge_2)

        # end of plugin constats

        split_peaks = strax.split_by_containment(peaks, events)
        split_peaks_ind = strax.fully_contained_in(peaks, events)
        result = np.zeros(len(peaks), self.dtype)
        drift_times = np.full(len(peaks), np.nan, dtype=np.float32)
        peaks_tags = np.full(
            len(peaks), "S2_not_associated_to_event", dtype="U50"
        )  # initial value for all peaks, array of peak tags to fill
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            mask = split_peaks_ind == event_i  # mask for peaks associated with the current event
            drift_times[mask] = sp["center_time"] - event["s1_center_time"]
            temp_peaks_tags_array = np.full(
                len(sp["area"]), "undefined", dtype="U50"
            )  # temporary array to hold the peak tags for the current event

            sp_S2_mask = sp["type"] == 2  # mask for S2 peaks associated with the current event

            # creating the physical S2 region mask
            # start S2 area>V3
            mask_S2_V3 = (
                (sp["area"] > v3)
                & (sp["range_50p_area"] > bottom_physical_horizontal_line_2)
                & (
                    sp["range_50p_area"]
                    < self.linear_line(sp["area"], v3, x_2_of_4spl, y_2_of_3spl, y_2_of_4spl)
                )
            )
            # end S2 area>V3
            # start V2<S2 area<V3
            mask_S2_V2 = (
                (sp["area"] <= v3)
                & (sp["area"] > x_points_2spl[2])
                & (sp["range_50p_area"] > bottom_physical_horizontal_line_1)
                & (
                    sp["range_50p_area"]
                    < self.linear_line(
                        sp["area"],
                        x_points_2spl[2],
                        v3,
                        np.polyval(coefficients_2spl, x_points_2spl[2]) - (y_2 - y_1),
                        y_2_of_3spl,
                    )
                )
            )
            # end V2<S2 area<V3
            # start V1<S2 area<V2
            mask_S2_V1 = (
                (sp["area"] > x_points_2spl[0])
                & (sp["area"] <= x_points_2spl[2])
                & (sp["range_50p_area"] > bottom_physical_horizontal_line_1)
                & (sp["range_50p_area"] < np.polyval(coefficients_2spl, sp["area"]))
            )
            # end V1<S2 area<V2
            # start S2 area<V1
            mask_S2_V0 = (
                (sp["area"] <= x_points_2spl[0])
                & (sp["area"] > x_points_1spl[0])
                & (sp["range_50p_area"] > bottom_physical_horizontal_line_1)
                & (sp["range_50p_area"] < np.polyval(coefficients_1spl, sp["area"]))
            )
            # end S2 area<V1
            mask_S2_physical_region = mask_S2_V0 | mask_S2_V1 | mask_S2_V2 | mask_S2_V3
            # end of creating the physical S2 region mask

            # creating the edge region mask
            # start edge region 2
            mask_in_edge_2 = (
                (sp["area"] <= stop_point_edge)
                & (sp["area"] > x_points_2spl[2])
                & (sp["range_50p_area"] > y_upper_for_edge)
                & (
                    sp["range_50p_area"]
                    < self.linear_line(
                        sp["area"],
                        x_points_2spl[2],
                        v3,
                        np.polyval(coefficients_2spl, x_points_2spl[2]) - (y_2 - y_1),
                        y_2_of_3spl,
                    )
                )
            )
            # end edge region 2
            # start edge region 1 and 0
            mask_in_edge_1 = (
                (sp["area"] > x_points_2spl[0])
                & (sp["area"] <= x_points_2spl[2])
                & (sp["range_50p_area"] > y_mid_for_edge)
                & (sp["range_50p_area"] < np.polyval(coefficients_2spl, sp["area"]))
            )
            # end edge region 1
            # start edge region 0
            mask_in_edge_0 = (
                (sp["area"] <= x_points_2spl[0])
                & (sp["area"] > x_points_1spl[0])
                & (sp["range_50p_area"] > y_mid_for_edge)
                & (sp["range_50p_area"] < np.polyval(coefficients_1spl, sp["area"]))
            )
            # end edge region 0
            mask_in_edge = mask_in_edge_0 | mask_in_edge_1 | mask_in_edge_2
            # end of creating the edge region mask

            # creating the e-train leakage mask based on the
            # seperation parameters and area/dt*length condition
            # Safe division with NaN/zero handling
            # Calculate denominators first to check for zeros/NaNs
            denom_1 = sp["endtime"] - sp["time"]
            denom_2 = sp["range_90p_area"]
            denom_3 = sp["endtime"] - sp["time"]
            denom_4 = sp["range_50p_area"]
            denom_var = sp["dt"] * sp["length"]

            # Use numpy.where to handle division by zero and NaN cases with error suppression
            # Returns NaN where numerator or denominator is zero/NaN, otherwise performs division
            with np.errstate(divide="ignore", invalid="ignore"):
                sep_1 = np.where(
                    (denom_1 != 0) & np.isfinite(denom_1) & np.isfinite(sp["range_50p_area"]),
                    sp["range_50p_area"] / denom_1,
                    np.nan,
                )
                sep_2 = np.where(
                    (denom_2 != 0) & np.isfinite(denom_2) & np.isfinite(sp["rise_time"]),
                    sp["rise_time"] / denom_2,
                    np.nan,
                )
                sep_3 = np.where(
                    (denom_3 != 0) & np.isfinite(denom_3) & np.isfinite(sp["range_90p_area"]),
                    sp["range_90p_area"] / denom_3,
                    np.nan,
                )
                sep_4 = np.where(
                    (denom_4 != 0) & np.isfinite(denom_4) & np.isfinite(sp["rise_time"]),
                    sp["rise_time"] / denom_4,
                    np.nan,
                )
                variable_to_check = np.where(
                    (denom_var != 0) & np.isfinite(denom_var) & np.isfinite(sp["area"]),
                    sp["area"] / denom_var,
                    np.nan,
                )

            # Ensure all separation parameters are finite (replace any remaining NaN/inf with NaN)
            sep_1 = np.where(np.isfinite(sep_1), sep_1, np.nan)
            sep_2 = np.where(np.isfinite(sep_2), sep_2, np.nan)
            sep_3 = np.where(np.isfinite(sep_3), sep_3, np.nan)
            sep_4 = np.where(np.isfinite(sep_4), sep_4, np.nan)
            variable_to_check = np.where(np.isfinite(variable_to_check), variable_to_check, np.nan)

            # Handle NaN values in mask calculations - NaN comparisons return False
            # This ensures mask keeps proper dimensions and NaN values are excluded
            mask_e_train_leakage_1 = (
                np.isfinite(sep_1)
                & (sep_1 > 0.33)
                & (sep_1 < 0.63)
                & np.isfinite(sep_2)
                & (sep_2 > 0.2)
                & (sep_2 < 0.63)
                & np.isfinite(sep_3)
                & (sep_3 > 0.75)
                & np.isfinite(sep_4)
                & (sep_4 > 0.42)
                & (sep_4 < 1.25)
            )
            mask_e_train_leakage_2 = np.isfinite(variable_to_check) & (variable_to_check < 0.3)

            mask_e_train_leakage = mask_e_train_leakage_1 & mask_e_train_leakage_2
            # end of creating the e-train leakage mask

            # do the peaks tagging per event
            temp_peaks_tags_array[(sp_S2_mask) & (mask_S2_physical_region) & (~mask_in_edge)] = (
                "S2_physical"
            )
            temp_peaks_tags_array[(sp_S2_mask) & (mask_in_edge) & (~mask_e_train_leakage)] = (
                "S2_physical"
            )
            temp_peaks_tags_array[(sp_S2_mask) & (mask_in_edge) & (mask_e_train_leakage)] = (
                "S2_etrain_leakage"
            )
            temp_peaks_tags_array[(sp_S2_mask) & (~mask_S2_physical_region)] = "S2_not_physical"
            peaks_tags[mask] = temp_peaks_tags_array

        # finalizing the peak tagging by adding the S1 and undefined tags for all of the peaks
        undefined_mask = peaks["type"] == 0
        peaks_tags[undefined_mask] = "undefined"
        mask_S1 = peaks["type"] == 1
        peaks_tags[mask_S1] = "S1"
        result["drift_time"] = drift_times
        result["peak_tag"] = peaks_tags
        result["time"] = peaks["time"]
        result["endtime"] = strax.endtime(peaks)
        return result
