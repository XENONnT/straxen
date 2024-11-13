import numpy as np
import strax
import straxen
from straxen.plugins.peaks._peak_positions_base import PeakPositionsBaseNT

export, __all__ = strax.exporter()


@export
class PeakPositionsCNF(PeakPositionsBaseNT):
    """Conditional Normalizing Flow for position reconstruction.

    This plugin reconstructs the position of S2 peaks using a conditional normalizing flow model.
    It provides x and y coordinates of the reconstructed position, along with uncertainty contours
    and uncertainty estimates in r and theta. For information on the model, see note_.

    .. _note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:juehang:flow_posrec_proposal_sr2  # noqa: E501

    Depends on: 'peaks'
    Provides: 'peak_positions_cnf'

    Configuration options:
    - min_reconstruction_area: Minimum area (PE) required for reconstruction
    - n_poly: Size of the uncertainty contour
    - sig: Confidence level of the contour
    - log_area_scale: Scaling parameter for log area
    - n_top_pmts: Number of top PMTs
    - pred_function: Path to the compiled JAX function for predictions

    """

    __version__ = "0.0.4"
    depends_on = "peaks"
    provides = "peak_positions_cnf"
    algorithm = "cnf"
    compressor = "zstd"
    parallel = True

    n_poly = straxen.URLConfig(
        default=16,
        infer_type=False,
        help="Size of uncertainty contour",
    )

    N_chunk_max = straxen.URLConfig(
        default=4096,
        infer_type=False,
        help="Maximum size of chunk for vectorised JAX function",
    )

    sig = straxen.URLConfig(
        default=0.393,
        infer_type=False,
        help="Confidence level of contour",
    )

    log_area_scale = straxen.URLConfig(
        default=10,
        infer_type=False,
        help="Scaling parameter for log area",
    )

    pred_function = straxen.URLConfig(
        default=(
            "jax://resource://flow_20240730.tar.gz?"
            "n_poly=plugin.n_poly&sig=plugin.sig&fmt=abs_path"
        ),
        help="Compiled JAX function",
    )

    def infer_dtype(self):
        """Define the data type for the output.

        Returns:
            dtype: Numpy dtype for the output array

        """
        dtype = [
            (
                (
                    f"Reconstructed {self.algorithm} S2 X position (cm), uncorrected",
                    f"x_{self.algorithm}",
                ),
                np.float32,
            ),
            (
                (
                    f"Reconstructed {self.algorithm} S2 Y position (cm), uncorrected",
                    f"y_{self.algorithm}",
                ),
                np.float32,
            ),
            (
                ("Position uncertainty contour", f"position_contours_{self.algorithm}"),
                np.float32,
                (self.n_poly + 1, 2),
            ),
            (("Position uncertainty in r (cm)", f"r_uncertainty_{self.algorithm}"), np.float32),
            (
                ("Position uncertainty in theta (rad)", f"theta_uncertainty_{self.algorithm}"),
                np.float32,
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def vectorized_prediction_chunk(self, flow_condition):
        """Compute predictions for a chunk of data.

        Args:
            flow_condition: Input data for the flow model

        Returns:
            xy: Predicted x and y coordinates
            contour: Uncertainty contours

        """
        N_entries = flow_condition.shape[0]
        if N_entries > self.N_chunk_max:
            raise ValueError("Chunk greater than max size")
        else:
            inputs = np.zeros((self.N_chunk_max, self.n_top_pmts + 1))
            inputs[:N_entries] = flow_condition
            xy, contour = self.pred_function(inputs)
            return xy[:N_entries], contour[:N_entries]

    def prediction_loop(self, flow_condition):
        """Compute predictions for arbitrary-size inputs using a loop.

        Args:
            flow_condition: Input data for the flow model

        Returns:
            xy: Predicted x and y coordinates
            contour: Uncertainty contours

        """
        N_entries = flow_condition.shape[0]
        if N_entries <= self.N_chunk_max:
            return self.vectorized_prediction_chunk(flow_condition)
        N_chunks = N_entries // self.N_chunk_max

        xy_list = []
        contour_list = []
        for i in range(N_chunks):
            xy, contour = self.vectorized_prediction_chunk(
                flow_condition[i * self.N_chunk_max : (i + 1) * self.N_chunk_max]
            )
            xy_list.append(xy)
            contour_list.append(contour)

        if N_chunks * self.N_chunk_max < N_entries:
            xy, contour = self.vectorized_prediction_chunk(
                flow_condition[(i + 1) * self.N_chunk_max :]
            )
            xy_list.append(xy)
            contour_list.append(contour)
        return np.concatenate(xy_list, axis=0), np.concatenate(contour_list, axis=0)

    def compute(self, peaks):
        """Compute the position reconstruction for the given peaks.

        Args:
            peaks: Input peak data

        Returns:
            result: Array with reconstructed positions and uncertainties

        """
        # Initialize result array
        result = np.ones(len(peaks), dtype=self.dtype)
        result["time"], result["endtime"] = peaks["time"], strax.endtime(peaks)

        # Set default values to NaN
        result[f"x_{self.algorithm}"] *= float("nan")
        result[f"y_{self.algorithm}"] *= float("nan")
        result[f"position_contours_{self.algorithm}"] *= float("nan")
        result[f"r_uncertainty_{self.algorithm}"] *= np.nan
        result[f"theta_uncertainty_{self.algorithm}"] *= np.nan

        # Keep large peaks only
        peak_mask = peaks["area"] > self.min_reconstruction_area
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result

        # Prepare input data for the flow model
        area_per_channel_top = peaks["area_per_channel"][peak_mask, 0 : self.n_top_pmts]
        total_top_areas = np.sum(area_per_channel_top, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            flow_data = np.concatenate(
                [
                    area_per_channel_top / total_top_areas[..., np.newaxis],
                    np.log(total_top_areas[..., np.newaxis]) / self.log_area_scale,
                ],
                axis=1,
            )

        # Get position reconstruction
        xy, contours = self.prediction_loop(flow_data)

        # Write output to the result array
        result[f"x_{self.algorithm}"][peak_mask] = xy[:, 0]
        result[f"y_{self.algorithm}"][peak_mask] = xy[:, 1]
        result[f"position_contours_{self.algorithm}"][peak_mask] = contours

        # Calculate uncertainties in r and theta
        r_array = np.linalg.norm(contours, axis=2)
        r_min = np.min(r_array, axis=1)
        r_max = np.max(r_array, axis=1)

        theta_array = np.arctan2(contours[..., 1], contours[..., 0])

        # Correction for circular nature of angle
        avg_theta = np.arctan2(
            np.mean(contours[..., 1], axis=1),
            np.mean(contours[..., 0], axis=1),
        )

        avg_theta = np.reshape(avg_theta, (avg_theta.shape[0], 1))
        theta_array_shift = (np.subtract(theta_array, avg_theta) + np.pi) % (2 * np.pi)
        theta_min = np.min(theta_array_shift, axis=1)
        theta_max = np.max(theta_array_shift, axis=1)

        theta_diff = theta_max - theta_min

        result[f"r_uncertainty_{self.algorithm}"][peak_mask] = (r_max - r_min) / 2
        result[f"theta_uncertainty_{self.algorithm}"][peak_mask] = np.abs(theta_diff) / 2

        return result
