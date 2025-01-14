import numpy as np
import numba
import strax
import straxen


export, __all__ = strax.exporter()


@export
class EventBasicsVanilla(strax.Plugin):
    """Computes the basic properties of the main/alternative S1/S2 within an event.

    The main S1 and alternative S1 are given by the largest two S1-Peaks within the event. The main
    S2 is given by the largest S2-Peak within the event, while alternative S2 is selected as the
    largest S2 other than main S2 in the time window [main S1 time, main S1 time + max drift time].

    """

    __version__ = "1.3.4"

    depends_on = ("events", "peak_basics", "peak_positions", "peak_proximity")
    provides = "event_basics"
    data_kind = "events"

    electron_drift_velocity = straxen.URLConfig(
        default="cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    allow_posts2_s1s = straxen.URLConfig(
        default=False,
        infer_type=False,
        help="Allow S1s past the main S2 to become the main S1 and S2",
    )

    force_main_before_alt = straxen.URLConfig(
        default=False,
        infer_type=False,
        help="Make the alternate S1 (and likewise S2) the main S1 if occurs before the main S1.",
    )

    force_alt_s2_in_max_drift_time = straxen.URLConfig(
        default=True,
        infer_type=False,
        help="Make sure alt_s2 is in max drift time starting from main S1",
    )

    event_s1_min_coincidence = straxen.URLConfig(
        default=2,
        infer_type=False,
        help=(
            "Event level S1 min coincidence. Should be >= s1_min_coincidence "
            "in the peaklet classification"
        ),
    )

    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z,
        infer_type=False,
        help="Total length of the TPC from the bottom of gate to the top of cathode wires [cm]",
    )

    def infer_dtype(self):
        # Basic event properties
        self._set_posrec_save()
        self._set_dtype_requirements()
        dtype = strax.time_fields + [
            ("n_peaks", np.int32, "Number of peaks in the event"),
            ("drift_time", np.float32, "Drift time between main S1 and S2 in ns"),
            ("event_number", np.int64, "Event number in this dataset"),
        ]

        dtype += self._get_si_dtypes(self.peak_properties)

        dtype += [
            (
                f"s1_max_diff",
                np.int32,
                f"Main S1 largest time difference between apexes of hits [ns]",
            ),
            (
                f"alt_s1_max_diff",
                np.int32,
                f"Alternate S1 largest time difference between apexes of hits [ns]",
            ),
            (
                f"s1_min_diff",
                np.int32,
                f"Main S1 smallest time difference between apexes of hits [ns]",
            ),
            (
                f"alt_s1_min_diff",
                np.int32,
                f"Alternate S1 smallest time difference between apexes of hits [ns]",
            ),
        ]

        dtype += [
            (f"s2_x", np.float32, f"Main S2 reconstructed X position, uncorrected [cm]"),
            (f"s2_y", np.float32, f"Main S2 reconstructed Y position, uncorrected [cm]"),
            (f"alt_s2_x", np.float32, f"Alternate S2 reconstructed X position, uncorrected [cm]"),
            (f"alt_s2_y", np.float32, f"Alternate S2 reconstructed Y position, uncorrected [cm]"),
            (f"area_before_main_s2", np.float32, f"Sum of areas before Main S2 [PE]"),
            (f"large_s2_before_main_s2", np.float32, f"The largest S2 before the Main S2 [PE]"),
        ]

        dtype += self._get_posrec_dtypes()
        return dtype

    def _set_dtype_requirements(self):
        """Needs to be run before inferring dtype as it is needed there."""
        # Properties to store for each peak (main and alternate S1 and S2)
        self.peak_properties = (
            ("time", np.int64, "start time since unix epoch [ns]"),
            ("center_time", np.int64, "weighted average center time since unix epoch [ns]"),
            ("median_time", np.float32, "weighted relative median time of the peak [ns]"),
            ("endtime", np.int64, "end time since unix epoch [ns]"),
            ("area", np.float32, "area, uncorrected [PE]"),
            ("n_channels", np.int16, "count of contributing PMTs"),
            ("n_hits", np.int16, "count of hits contributing at least one sample to the peak"),
            ("n_competing", np.int32, "number of competing peaks"),
            ("max_pmt", np.int16, "PMT number which contributes the most PE"),
            ("max_pmt_area", np.float32, "area in the largest-contributing PMT (PE)"),
            ("range_50p_area", np.float32, "width, 50% area [ns]"),
            ("range_90p_area", np.float32, "width, 90% area [ns]"),
            ("rise_time", np.float32, "time between 10% and 50% area quantiles [ns]"),
            ("area_fraction_top", np.float32, "fraction of area seen by the top PMT array"),
            ("tight_coincidence", np.int16, "channel within tight range of mean"),
            ("n_saturated_channels", np.int16, "total number of saturated channels"),
        )

    def setup(self):
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)

    @staticmethod
    def _get_si_dtypes(peak_properties):
        """Get properties for S1/S2 from peaks directly."""
        si_dtype = []
        for s_i in [1, 2]:
            # Peak indices
            si_dtype += [
                (f"s{s_i}_index", np.int32, f"Main S{s_i} peak index in event"),
                (f"alt_s{s_i}_index", np.int32, f"Alternate S{s_i} peak index in event"),
            ]

            # Peak properties
            for name, dt, comment in peak_properties:
                si_dtype += [
                    (f"s{s_i}_{name}", dt, f"Main S{s_i} {comment}"),
                    (f"alt_s{s_i}_{name}", dt, f"Alternate S{s_i} {comment}"),
                ]

            # Drifts and delays
            si_dtype += [
                (
                    f"alt_s{s_i}_interaction_drift_time",
                    np.float32,
                    f"Drift time using alternate S{s_i} [ns]",
                ),
                (f"alt_s{s_i}_delay", np.int32, f"Time between main and alternate S{s_i} [ns]"),
            ]
        return si_dtype

    def _set_posrec_save(self):
        """Parse x_mlp et cetera if needed to get the algorithms used and set required class
        attributes."""
        posrec_fields = self.deps["peak_positions"].dtype_for("peak_positions").names
        posrec_names = [d.split("_")[-1] for d in posrec_fields if "x_" in d]

        # Preserve order. "set" is not ordered and dtypes should always be ordered
        self.pos_rec_labels = list(set(posrec_names))
        self.pos_rec_labels.sort()

        self.posrec_save = [(xy + alg) for xy in ["x_", "y_"] for alg in self.pos_rec_labels]

    def _get_posrec_dtypes(self):
        """Get S2 positions for each of the position reconstruction algorithms."""
        posrec_dtpye = []

        for alg in self.pos_rec_labels:
            # S2 positions
            posrec_dtpye += [
                (
                    f"s2_x_{alg}",
                    np.float32,
                    f"Main S2 {alg}-reconstructed X position, uncorrected [cm]",
                ),
                (
                    f"s2_y_{alg}",
                    np.float32,
                    f"Main S2 {alg}-reconstructed Y position, uncorrected [cm]",
                ),
                (
                    f"alt_s2_x_{alg}",
                    np.float32,
                    f"Alternate S2 {alg}-reconstructed X position, uncorrected [cm]",
                ),
                (
                    f"alt_s2_y_{alg}",
                    np.float32,
                    f"Alternate S2 {alg}-reconstructed Y position, uncorrected [cm]",
                ),
            ]

        return posrec_dtpye

    def compute(self, events, peaks):
        result = np.zeros(len(events), dtype=self.dtype)
        strax.set_nan_defaults(result)

        split_peaks = strax.split_by_containment(peaks, events)

        result["time"] = events["time"]
        result["endtime"] = events["endtime"]
        result["event_number"] = events["event_number"]

        self.fill_events(result, split_peaks)
        return result

    # If copy_largest_peaks_into_event is ever numbafied, also numbafy this function
    def fill_events(self, result_buffer, split_peaks):
        """Loop over the events and peaks within that event."""
        for event_i, peaks_in_event_i in enumerate(split_peaks):
            n_peaks = len(peaks_in_event_i)
            result_buffer[event_i]["n_peaks"] = n_peaks

            if not n_peaks:
                raise ValueError(f"No peaks within event {event_i}?")

            self.fill_result_i(result_buffer[event_i], peaks_in_event_i)

    def fill_result_i(self, event, peaks):
        """For a single event with the result_buffer."""
        # Consider S2s first, then S1s (to enable allow_posts2_s1s = False)
        # number_of_peaks=0 selects all available s2 and sort by area
        largest_s2s, s2_idx = self.get_largest_sx_peaks(peaks, s_i=2, number_of_peaks=0)

        if not self.allow_posts2_s1s and len(largest_s2s):
            s1_latest_time = largest_s2s[0]["time"]
        else:
            s1_latest_time = np.inf

        largest_s1s, s1_idx = self.get_largest_sx_peaks(
            peaks,
            s_i=1,
            s1_before_time=s1_latest_time,
            s1_min_coincidence=self.event_s1_min_coincidence,
        )

        if self.force_alt_s2_in_max_drift_time:
            s2_idx, largest_s2s = self.find_main_alt_s2(
                largest_s1s,
                s2_idx,
                largest_s2s,
                self.drift_time_max,
            )
        else:
            # Select only the largest two S2s
            largest_s2s, s2_idx = largest_s2s[0:2], s2_idx[0:2]

        if self.force_main_before_alt:
            s2_order = strax.stable_argsort(largest_s2s["time"])
            largest_s2s = largest_s2s[s2_order]
            s2_idx = s2_idx[s2_order]

        self.set_sx_index(event, s1_idx, s2_idx)
        self.set_event_properties(event, largest_s1s, largest_s2s, peaks)

        # Loop over S1s and S2s and over main / alt.
        for s_i, largest_s_i in enumerate([largest_s1s, largest_s2s], 1):
            # Largest index 0 -> main sx, 1 -> alt sx
            for largest_index, main_or_alt in enumerate(["s", "alt_s"]):
                peak_properties_to_save = [name for name, _, _ in self.peak_properties]
                if s_i == 1:
                    peak_properties_to_save += ["max_diff", "min_diff"]
                elif s_i == 2:
                    peak_properties_to_save += ["x", "y"]
                    peak_properties_to_save += self.posrec_save
                field_names = [f"{main_or_alt}{s_i}_{name}" for name in peak_properties_to_save]
                self.copy_largest_peaks_into_event(
                    event, largest_s_i, largest_index, field_names, peak_properties_to_save
                )

    @staticmethod
    @numba.njit
    def find_main_alt_s2(largest_s1s, s2_idx, largest_s2s, drift_time_max):
        """Require alt_s2 happens between main S1 and maximum drift time."""
        if len(largest_s1s) > 0 and len(largest_s2s) > 1:
            # If there is a valid s1-s2 pair and has a second s2, then check alt s2 validity
            s2_after_s1 = largest_s2s["center_time"] > largest_s1s[0]["center_time"]
            s2_before_max_drift_time = (
                largest_s2s["center_time"] - largest_s1s[0]["center_time"]
            ) < 1.01 * drift_time_max
            mask = s2_after_s1 & s2_before_max_drift_time
            # The selection avoids main_S2
            mask[0] = True
            # Take main and the largest valid alt_S2
            s2_idx, largest_s2s = s2_idx[mask], largest_s2s[mask]
        return s2_idx[:2], largest_s2s[:2]

    @staticmethod
    @numba.njit
    def set_event_properties(result, largest_s1s, largest_s2s, peaks):
        """Get properties like drift time and area before main S2."""
        # Compute drift times only if we have a valid S1-S2 pair
        if len(largest_s1s) > 0 and len(largest_s2s) > 0:
            result["drift_time"] = largest_s2s[0]["center_time"] - largest_s1s[0]["center_time"]
            if len(largest_s1s) > 1:
                result["alt_s1_interaction_drift_time"] = (
                    largest_s2s[0]["center_time"] - largest_s1s[1]["center_time"]
                )
                result["alt_s1_delay"] = (
                    largest_s1s[1]["center_time"] - largest_s1s[0]["center_time"]
                )
            if len(largest_s2s) > 1:
                result["alt_s2_interaction_drift_time"] = (
                    largest_s2s[1]["center_time"] - largest_s1s[0]["center_time"]
                )
                result["alt_s2_delay"] = (
                    largest_s2s[1]["center_time"] - largest_s2s[0]["center_time"]
                )

        # areas before main S2
        if len(largest_s2s):
            peaks_before_ms2 = peaks[peaks["time"] < largest_s2s[0]["time"]]
            result["area_before_main_s2"] = np.sum(peaks_before_ms2["area"])

            s2peaks_before_ms2 = peaks_before_ms2[peaks_before_ms2["type"] == 2]
            if len(s2peaks_before_ms2) == 0:
                result["large_s2_before_main_s2"] = 0
            else:
                result["large_s2_before_main_s2"] = np.max(s2peaks_before_ms2["area"])
        return result

    @staticmethod
    # @numba.njit <- works but slows if fill_events is not numbafied
    def get_largest_sx_peaks(
        peaks, s_i, s1_before_time=np.inf, s1_min_coincidence=0, number_of_peaks=2
    ):
        """Get the largest S1/S2.

        For S1s allow a min coincidence and max time

        """
        # Find all peaks of this type (S1 or S2)
        s_mask = peaks["type"] == s_i
        if s_i == 1:
            s_mask &= peaks["time"] < s1_before_time
            s_mask &= peaks["tight_coincidence"] >= s1_min_coincidence

        selected_peaks = peaks[s_mask]
        s_index = np.arange(len(peaks))[s_mask]
        largest_peaks = strax.stable_argsort(selected_peaks["area"])[-number_of_peaks:][::-1]
        return selected_peaks[largest_peaks], s_index[largest_peaks]

    # If only we could numbafy this... Unfortunatly we cannot.
    # Perhaps we could one day consider doing something like strax.copy_to_buffer
    @staticmethod
    def copy_largest_peaks_into_event(
        result,
        largest_s_i,
        main_or_alt_index,
        result_fields,
        peak_fields,
    ):
        """For one event, write all the peak_fields (e.g. "area") of the peak (largest_s_i) into
        their associated field in the event (e.g. s1_area), main_or_alt_index differentiates between
        main (index 0) and alt (index 1)"""
        index_not_in_list_of_largest_peaks = main_or_alt_index >= len(largest_s_i)
        if index_not_in_list_of_largest_peaks:
            # There is no such peak. E.g. main_or_alt_index == 1 but largest_s_i = ["Main S1"]
            # Asking for index 1 doesn't work on a len 1 list of peaks.
            return

        for i, ev_field in enumerate(result_fields):
            p_field = peak_fields[i]
            if p_field not in ev_field:
                raise ValueError("Event fields must derive from the peak fields")
            result[ev_field] = largest_s_i[main_or_alt_index][p_field]

    @staticmethod
    # @numba.njit <- works but slows if fill_events is not numbafied
    def set_sx_index(res, s1_idx, s2_idx):
        if len(s1_idx):
            res["s1_index"] = s1_idx[0]
            if len(s1_idx) > 1:
                res["alt_s1_index"] = s1_idx[1]
        if len(s2_idx):
            res["s2_index"] = s2_idx[0]
            if len(s2_idx) > 1:
                res["alt_s2_index"] = s2_idx[1]
