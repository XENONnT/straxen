"""Position reconstruction for Xenon-nT."""

from typing import Optional

import numpy as np
import strax
import straxen
from warnings import warn

export, __all__ = strax.exporter()

DEFAULT_POSREC_ALGO = "mlp"


@export
class PeakPositionsBaseNT(strax.Plugin):
    """Base class for reconstructions.

    This class should only be used when subclassed for the different algorithms. Provides
    x_algorithm, y_algorithm for all peaks > than min-reconstruction area based on the top array.

    """

    __version__ = "0.0.0"

    depends_on = "peaks"
    algorithm: Optional[str] = None
    compressor = "zstd"
    parallel = True

    min_reconstruction_area = straxen.URLConfig(
        help="Skip reconstruction if area (PE) is less than this",
        default=10,
        infer_type=False,
    )

    n_top_pmts = straxen.URLConfig(
        default=straxen.n_top_pmts, infer_type=False, help="Number of top PMTs"
    )

    def infer_dtype(self):
        if self.algorithm is None:
            raise NotImplementedError(
                f"Base class should not be used without algorithm as done in {__class__.__name__}"
            )
        dtype = [
            (
                "x_" + self.algorithm,
                np.float32,
                f"Reconstructed {self.algorithm} S2 X position (cm), uncorrected",
            ),
            (
                "y_" + self.algorithm,
                np.float32,
                f"Reconstructed {self.algorithm} S2 Y position (cm), uncorrected",
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def get_tf_model(self):
        """Simple wrapper to have several tf_model_mlp, tf_model_cnf, ..

        point to this same function in the compute method

        """
        model = getattr(self, f"tf_model_{self.algorithm}", None)
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
        result = np.ones(len(peaks), dtype=self.dtype)
        result["time"], result["endtime"] = peaks["time"], strax.endtime(peaks)

        result["x_" + self.algorithm] *= np.nan
        result["y_" + self.algorithm] *= np.nan
        model = self.get_tf_model()

        if model is None:
            # This plugin is disabled since no model is provided
            return result

        # Keep large peaks only
        peak_mask = peaks["area"] > self.min_reconstruction_area
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result

        # Getting actual position reconstruction
        area_per_channel_top = peaks["area_per_channel"][peak_mask, 0 : self.n_top_pmts]
        with np.errstate(divide="ignore", invalid="ignore"):
            area_per_channel_top = area_per_channel_top / np.max(
                area_per_channel_top, axis=1
            ).reshape(-1, 1)
        area_per_channel_top = area_per_channel_top.reshape(-1, self.n_top_pmts)
        output = model.predict(area_per_channel_top, verbose=0)

        # writing output to the result
        result["x_" + self.algorithm][peak_mask] = output[:, 0]
        result["y_" + self.algorithm][peak_mask] = output[:, 1]
        return result
