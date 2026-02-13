"""Peak-only CNF position reconstruction (no peaklet dependencies).

This plugin reconstructs S2 positions directly from peak-level area_per_channel,
without requiring peaklet-level position reconstruction. This significantly reduces
RAM usage during online processing.

This uses the same CNF (Conditional Normalizing Flow) algorithm as PeakletPositionsCNF,
but operates on merged peaks instead of individual peaklets. Based on the v2.2.7
approach of computing positions after S2 merging.

Trade-offs:
- Pro: ~27 GB less RAM (no peaklet waveform loading)
- Con: Slightly less accurate positions (no peaklet-level information)
- Con: Less effective S2 merging (time-only, not time+position)

Use this for online DAQ where RAM is constrained. For offline analysis,
use PeakPositionsCNF which provides better accuracy via peaklet positions.
"""

import numpy as np
import numba
import strax
import straxen

export, __all__ = strax.exporter()


@export
class PeakPositionsCNFPeakOnly(strax.Plugin):
    """CNF position reconstruction from peak area_per_channel (no peaklet dependencies).
    
    This is a memory-efficient alternative to PeakPositionsCNF that computes
    positions after S2 merging, not before. Uses the same CNF algorithm but
    operates on peak-level data.
    """

    __version__ = "0.0.0"
    provides = "peak_positions_cnf"
    depends_on = "peaks"
    algorithm = "cnf"
    child_plugin = True
    compressor = "zstd"
    parallel = True

    min_reconstruction_area = straxen.URLConfig(
        help="Skip reconstruction if area (PE) is less than this",
        default=0,
        infer_type=False,
    )

    n_top_pmts = straxen.URLConfig(
        default=straxen.n_top_pmts, infer_type=False, help="Number of top PMTs"
    )

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

    cnf_pred_function = straxen.URLConfig(
        default=(
            "jax://resource://xedocs://posrec_models"
            "?kind=cnf&attr=value&n_poly=plugin.n_poly&sig=plugin.sig&fmt=abs_path"
            "&version=ONLINE&run_id=plugin.run_id"
        ),
        help="Compiled JAX function",
    )

    @staticmethod
    def calculate_theta_diff(theta_array, avg_theta):
        """Calculate the difference between maximum and minimum angles from an array of angles.
        
        Normalizes angular difference into range [0, 2π).
        """
        # Correction to handle circular nature of angles
        theta_array_shift = (theta_array - avg_theta[..., np.newaxis] + np.pi) % (2 * np.pi)
        theta_min = np.min(theta_array_shift, axis=1)
        theta_max = np.max(theta_array_shift, axis=1)
        theta_diff = theta_max - theta_min
        return theta_diff

    def infer_dtype(self):
        """Define the data type for the output."""
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
                ("Position uncertainty contour (cm)", f"position_contour_{self.algorithm}"),
                np.float32,
                (self.n_poly + 1, 2),
            ),
            (
                (
                    "Area in position uncertainty contour (cm^2)",
                    f"position_contour_area_{self.algorithm}",
                ),
                np.float32,
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
            xy, contour = self.cnf_pred_function(inputs)
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

    @staticmethod
    @numba.njit(cache=True, nogil=True)
    def polygon_area(polygon):
        """Calculate and return the area of a polygon.

        The input is a 3D numpy array where the first dimension represents individual polygons,
        the second dimension represents vertices, and the third dimension represents x and y
        coordinates of each vertex.
        """
        x = polygon[..., 0]
        y = polygon[..., 1]
        result = np.zeros(polygon.shape[0], dtype=np.float32)
        for i in range(x.shape[-1]):
            result += (x[..., i] * y[..., i - 1]) - (y[..., i] * x[..., i - 1])
        return 0.5 * np.abs(result)

    def compute(self, peaks):
        """Compute CNF position reconstruction from peak area_per_channel.
        
        This operates on merged peaks, using aggregated area_per_channel data
        rather than individual peaklet waveforms.
        """
        # Initialize result array
        result = np.ones(len(peaks), dtype=self.dtype)
        # Set default values to NaN
        strax.set_nan_defaults(result)
        result["time"], result["endtime"] = peaks["time"], strax.endtime(peaks)

        # Keep large peaks only
        peak_mask = peaks["area"] > self.min_reconstruction_area
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result

        # Prepare input data for the CNF model
        # This uses the same approach as PeakletPositionsCNF but on peak data
        area_per_channel_top = peaks["area_per_channel"][peak_mask, : self.n_top_pmts]
        total_top_areas = np.sum(area_per_channel_top, axis=1)
        
        # Create flow condition: normalized PMT pattern + log(area)
        with np.errstate(divide="ignore", invalid="ignore"):
            flow_data = np.concatenate(
                [
                    area_per_channel_top / total_top_areas[..., np.newaxis],
                    np.log(total_top_areas[..., np.newaxis]) / self.log_area_scale,
                ],
                axis=1,
            )

        # Get position reconstruction from CNF
        xy, contours = self.prediction_loop(flow_data)

        # Write output to the result array
        result[f"x_{self.algorithm}"][peak_mask] = xy[:, 0]
        result[f"y_{self.algorithm}"][peak_mask] = xy[:, 1]
        result[f"position_contour_{self.algorithm}"][peak_mask] = contours
        result[f"position_contour_area_{self.algorithm}"][peak_mask] = self.polygon_area(
            result[f"position_contour_{self.algorithm}"][peak_mask]
        )

        # Calculate uncertainties in r and theta
        r_array = np.linalg.norm(contours, axis=2)
        r_min = np.min(r_array, axis=1)
        r_max = np.max(r_array, axis=1)

        theta_array = np.arctan2(contours[..., 1], contours[..., 0])
        avg_theta = np.arctan2(xy[:, 1], xy[:, 0])
        theta_diff = self.calculate_theta_diff(theta_array, avg_theta)

        result[f"r_uncertainty_{self.algorithm}"][peak_mask] = (r_max - r_min) / 2
        result[f"theta_uncertainty_{self.algorithm}"][peak_mask] = np.abs(theta_diff) / 2

        return result
