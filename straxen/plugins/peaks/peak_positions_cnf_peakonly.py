"""Peak-only CNF position reconstruction (no peaklet dependencies).

This plugin reconstructs S2 positions directly from peak-level area_per_channel,
without requiring peaklet-level position reconstruction. This significantly reduces
RAM usage during online processing.

This is based on the v2.2.7 approach (PeakPositionsBaseNT) which computed positions
after S2 merging, rather than before merging (as done in PR #1482).

Trade-offs:
- Pro: ~27 GB less RAM (no peaklet waveform loading)
- Con: Slightly less accurate positions (no peaklet-level information)
- Con: Less effective S2 merging (time-only, not time+position)

Use this for online DAQ where RAM is constrained. For offline analysis,
use PeakPositionsCNF which provides better accuracy via peaklet positions.
"""

from typing import Optional
from warnings import warn

import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@export
class PeakPositionsCNFPeakOnly(strax.Plugin):
    """CNF position reconstruction from peak area_per_channel (no peaklet dependencies).
    
    This is a memory-efficient alternative to PeakPositionsCNF that computes
    positions after S2 merging, not before. Based on the v2.2.7 approach.
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

    tf_model_cnf = straxen.URLConfig(
        default=(
            "tf://"
            "resource://"
            "xedocs://posrec_models"
            "?version=ONLINE"
            "&run_id=plugin.run_id"
            "&fmt=abs_path"
            "&kind=cnf"
            "&attr=value"
        ),
        help=(
            'CNF model. Should be opened using the "tf" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )

    def infer_dtype(self):
        dtype = [
            (
                "x_cnf",
                np.float32,
                "Reconstructed cnf S2 X position (cm), uncorrected",
            ),
            (
                "y_cnf",
                np.float32,
                "Reconstructed cnf S2 Y position (cm), uncorrected",
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def get_tf_model(self):
        """Get the TensorFlow model for CNF position reconstruction."""
        model = self.tf_model_cnf
        if model is None:
            warn(
                f"Setting model to None for {self.__class__.__name__} will "
                f"set only nans as output for {self.algorithm}"
            )
        if isinstance(model, str):
            raise ValueError(
                f"open files from tf:// protocol! Got {model} "
                "instead, see tests/test_posrec.py for examples."
            )
        return model

    def compute(self, peaks):
        """Compute CNF positions from peak area_per_channel.
        
        This is the v2.2.7 approach: reconstruct positions from the merged peak's
        area_per_channel, not from individual peaklet waveforms.
        """
        result = np.ones(len(peaks), dtype=self.dtype)
        result["time"], result["endtime"] = peaks["time"], strax.endtime(peaks)

        result["x_cnf"] *= float("nan")
        result["y_cnf"] *= float("nan")
        
        model = self.get_tf_model()
        if model is None:
            # This plugin is disabled since no model is provided
            return result

        # Keep large peaks only
        peak_mask = peaks["area"] > self.min_reconstruction_area
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result

        # Getting actual position reconstruction from area_per_channel
        # This is aggregated data from the merged peak, no waveforms needed
        area_per_channel_top = peaks["area_per_channel"][peak_mask, 0 : self.n_top_pmts]
        
        # Normalize per peak
        with np.errstate(divide="ignore", invalid="ignore"):
            area_per_channel_top = area_per_channel_top / np.max(
                area_per_channel_top, axis=1
            ).reshape(-1, 1)
        area_per_channel_top = area_per_channel_top.reshape(-1, self.n_top_pmts)
        
        # Run neural network
        output = model.predict(area_per_channel_top, verbose=0)

        # Write output to the result
        result["x_cnf"][peak_mask] = output[:, 0]
        result["y_cnf"][peak_mask] = output[:, 1]
        
        return result
