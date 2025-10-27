import os
import numpy as np
import torch

import strax
import straxen
from .peak_ambience import _quick_assign


class PeakPatternFit(strax.Plugin):
    __version__ = "0.0.0"
    depends_on = ("peaks", "peak_positions")
    provides = "peak_pattern_fit"

    dtype = strax.time_fields + [
        (
            ("Data-driven based likelihood value for peak", "neural_2llh"),
            np.float32,
        ),
    ]

    n_top_pmts = straxen.URLConfig(type=int, help="Number of top TPC PMTs")

    s2_tf_model = straxen.URLConfig(
        help="S2 (x, y) optical data-driven model",
        infer_type=False,
        default=(
            "tf://resource://"
            "XENONnT_s2_optical_map_data_driven_ML_v0_2021_11_25.keras.tar.gz"
            "?custom_objects=plugin.s2_map_custom_objects"
            "&fmt=abs_path&register=True"
        ),
    )

    s2_min_area_pattern_fit = straxen.URLConfig(
        infer_type=False,
        help="Skip EventPatternFit reconstruction if S2 area (PE) is less than this",
        default=10,
    )

    @property
    def s2_map_custom_objects(self):
        def _logl_loss(patterns_true, likelihood):
            return likelihood / 10.0

        return {"_logl_loss": _logl_loss}

    def setup(self):
        # Getting S2 data-driven tensorflow models
        self.model = self.s2_tf_model

        import tensorflow as tf

        self.model_chi2 = tf.keras.Model(
            self.model.inputs, self.model.get_layer("Likelihood").output
        )

    def compute(self, peaks):
        result = np.zeros(len(peaks), dtype=self.dtype)
        result["time"] = peaks["time"]
        result["endtime"] = strax.endtime(peaks)

        x, y = peaks["x"], peaks["y"]
        s2_mask = peaks["type"] == 2
        s2_mask &= peaks["area"] > self.s2_min_area_pattern_fit
        s2_mask &= peaks["area_fraction_top"] > 0

        # default value is nan, it will be ovewrite if the event satisfy the requirements
        result["neural_2llh"] = np.nan

        # Produce position and top pattern to feed tensorflow model, return chi2/N
        if np.sum(s2_mask):
            s2_pos = np.stack((x, y)).T[s2_mask]
            s2_pat = peaks["area_per_channel"][s2_mask, : self.n_top_pmts]
            # Output[0]: loss function, -2*log-likelihood, Output[1]: chi2
            result["neural_2llh"][s2_mask] = self.model_chi2.predict(
                {"xx": s2_pos, "yy": s2_pat}, verbose=0
            )[1]

        return result


class PeakCNFScore(strax.OverlapWindowPlugin):
    __version__ = "0.0.0"
    depends_on = ("peak_pattern_fit", "peak_basics", "peak_positions")
    provides = "peak_cnf_score"
    save_when = strax.SaveWhen.EXPLICIT

    cnf_batch_size = straxen.URLConfig(
        default=1000,
        type=int,
        track=True,
        help="Batch size for vectorized CNF evaluation",
    )

    cnf_time_window_backward = straxen.URLConfig(
        default=int(1e9),
        type=int,
        track=True,
        help="Search for peaks in this time window [ns]",
    )

    cnf_besearched_peak_minimum_area = straxen.URLConfig(
        default=1e4,
        type=float,
        track=True,
        help="Mininum area to cast the CNF scores",
    )

    cnf_model_folder = straxen.URLConfig(
        default="/project2/lgrandi/s2_only/model",
        cache=True,
        help="CNF model for isolated S2 rejection",
    )

    pi_rate_path = straxen.URLConfig(
        default="selection://plugin.sr?"
        "sr0=/project2/lgrandi/dali/shenyangshi/run_selection/sr0/result.npy&"
        "sr1=/project2/lgrandi/dali/shenyangshi/run_selection/sr1/result.npy&"
        "sr2=/project2/lgrandi/dali/shenyangshi/run_selection/sr2/result.npy&",
        cache=True,
        help="Photoionization file for each run that is loadable",
    )

    sr = straxen.URLConfig(
        default="science_run://plugin.run_id?&phase=False",
        help="Science run to be used for the cut. It can affect the parameters of the cut.",
    )

    cnf_event_upper_s2_area = straxen.URLConfig(
        default=1e3,
        type=float,
        track=True,
        help="Upper S2 area for the event to be considered",
    )

    def get_window_size(self):
        # This method is required by the OverlapWindowPlugin class
        return (10 * self.cnf_time_window_backward, 0)

    def get_cnf_model_path(self):
        """If the run is even, use the odd model, and vice versa.

        This is for CNF simulation and score computation.

        """
        if int(self.run_id) % 2 == 0:
            parity = "odd"
        elif int(self.run_id) % 2 == 1:
            parity = "even"
        else:
            raise ValueError("The run ID must be an integer. Please check the run ID.")
        path = os.path.join(self.cnf_model_folder, f"model_combined_{self.sr}_{parity}_0515.pt")
        return path

    def setup(self):
        self.cnf_model_path = self.get_cnf_model_path()
        self.cnf_model = torch.load(self.cnf_model_path, weights_only=False)
        self.pi_rate_file = np.load(self.pi_rate_path, allow_pickle=True)
        if np.int64(self.run_id) not in np.int64(self.pi_rate_file["run_id"]):
            raise ValueError(
                f"Run {self.run_id} not found in the photoionization rate file. "
                f"Please check the run ID and the file."
            )
        else:
            self.pi_rate = self.pi_rate_file["pi_rate"][
                np.int64(self.pi_rate_file["run_id"]) == np.int64(self.run_id)
            ][0]

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (("Maximum conditional normalizing flow score", "cnf_score"), np.float32),
            (("Time difference to the S2 casting maximum CNF score", "cnf_nearest_dt"), np.int64),
            (("S2 area of the S2 casting maximum CNF score", "cnf_nearest_s2_area"), np.float32),
            (
                ("Position difference to the S2 casting maximum CNF score", "cnf_nearest_dr"),
                np.float32,
            ),
        ]
        return dtype

    def compute(self, peaks):
        argsort = strax.stable_argsort(peaks["center_time"])
        _peaks = peaks[argsort].copy()
        result = np.zeros(len(peaks), self.dtype)
        _quick_assign(argsort, result, self.compute_cnf_score(peaks, _peaks))
        return result

    def compute_cnf_score(self, peaks, current_peak):
        # energetic peaks as before
        mask_peaks = peaks["type"] == 2
        mask_peaks &= peaks["area"] >= self.cnf_besearched_peak_minimum_area

        # only keep S2 peaks with area < cnf_event_upper_s2_area
        small_mask = current_peak["type"] == 2
        small_mask &= current_peak["area"] < self.cnf_event_upper_s2_area

        # prepare empty result for all peaks
        result = np.zeros(len(current_peak), dtype=self.dtype)
        strax.set_nan_defaults(result)

        if small_mask.any():
            # compute CNF only for them
            small_results = self._compute_cnf_score(peaks[mask_peaks], current_peak[small_mask])

            # write back into the full array
            result[small_mask] = small_results

        # fill time/endtime for completeness
        result["time"] = current_peak["time"]
        result["endtime"] = strax.endtime(current_peak)

        return result

    def _compute_cnf_score(self, peaks, current_peaks):
        """Compute conditional normalizing flow scores using batch processing."""
        # Create the output array.
        result = np.zeros(len(current_peaks), dtype=self.dtype)
        # Define the region-of-interest time window for each event.
        roi_cnf = np.zeros(len(current_peaks), dtype=strax.time_fields)
        roi_cnf["time"] = current_peaks["center_time"] - self.cnf_time_window_backward
        roi_cnf["endtime"] = current_peaks["center_time"]

        # Use strax.touching_windows to find for each event the index range in the peaks array.
        split_peaks = strax.touching_windows(peaks, roi_cnf)

        n_peaks = len(current_peaks)
        batch_size = self.cnf_batch_size

        # Process the peaks in batches.
        for batch_start in range(0, n_peaks, batch_size):
            batch_end = min(batch_start + batch_size, n_peaks)
            batch_current_peaks = current_peaks[batch_start:batch_end]
            batch_split_peaks = split_peaks[batch_start:batch_end]

            batch_results = self._vectorized_cnf_scores(
                peaks, batch_current_peaks, batch_split_peaks
            )
            result[batch_start:batch_end] = batch_results

        result["time"] = current_peaks["time"]
        result["endtime"] = strax.endtime(current_peaks)

        return result

    def _vectorized_cnf_scores(self, peaks, current_peaks, split_peaks):
        batch_size = len(current_peaks)
        batch_result = np.zeros(batch_size, dtype=self.dtype)

        c_batch = []
        x_batch = []
        c_size = []

        for idx, window in enumerate(split_peaks):
            casting_peaks = peaks[window[0] : window[1]]
            # Conditions

            if len(casting_peaks) == 0:
                c_size.append(0)
                c_batch.append(np.empty((0, 4), dtype=np.float32))
                x_batch.append(np.empty((0, 9), dtype=np.float32))
                continue

            c_size.append(len(casting_peaks))

            c_obs = np.stack(
                [
                    casting_peaks["area"],
                    casting_peaks["x"],
                    casting_peaks["y"],
                    np.ones(len(casting_peaks)) * self.pi_rate,
                ],
                axis=1,
            )
            # Small S2s
            x_obs = np.stack(
                [
                    current_peaks[idx]["center_time"] - casting_peaks["time"],
                    np.full(len(casting_peaks), current_peaks[idx]["area"]),
                    np.full(len(casting_peaks), current_peaks[idx]["x"]),
                    np.full(len(casting_peaks), current_peaks[idx]["y"]),
                    np.full(len(casting_peaks), current_peaks[idx]["range_50p_area"]),
                    np.full(len(casting_peaks), current_peaks[idx]["range_90p_area"]),
                    np.full(len(casting_peaks), current_peaks[idx]["rise_time"]),
                    np.full(len(casting_peaks), current_peaks[idx]["area_fraction_top"]),
                    np.full(len(casting_peaks), current_peaks[idx]["neural_2llh"]),
                ],
                axis=1,
            )

            # Append the results to the batch lists
            c_batch.append(c_obs)
            x_batch.append(x_obs)

        if sum(c_size) == 0:
            return batch_result

        c_batch = np.concatenate(c_batch, axis=0)
        x_batch = np.concatenate(x_batch, axis=0)
        log_pxc = self.cnf_model.evaluate(
            torch.tensor(c_batch, dtype=torch.float32),
            torch.tensor(x_batch, dtype=torch.float32),
            combine_rate=True,
        ).numpy()
        pdf = np.exp(log_pxc)

        offset = 0
        for i in range(batch_size):
            n_cand = c_size[i]
            if n_cand == 0:
                offset += 0
                continue
            pdf_i = pdf[offset : offset + n_cand]
            offset += n_cand

            best_idx = np.argmax(pdf_i)
            casting_peaks = peaks[split_peaks[i][0] : split_peaks[i][1]]

            batch_result[i]["cnf_score"] = pdf_i[best_idx]
            batch_result[i]["cnf_nearest_dt"] = (
                current_peaks[i]["center_time"] - casting_peaks["time"][best_idx]
            )
            batch_result[i]["cnf_nearest_s2_area"] = casting_peaks["area"][best_idx]
            batch_result[i]["cnf_nearest_dr"] = np.sqrt(
                (current_peaks[i]["x"] - casting_peaks["x"][best_idx]) ** 2
                + (current_peaks[i]["y"] - casting_peaks["y"][best_idx]) ** 2
            )

        return batch_result
