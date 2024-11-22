from typing import Optional

import strax
import straxen
import numpy as np
from warnings import warn

export, __all__ = strax.exporter()


@export
class EventS2PositionBase(strax.Plugin):
    """Base pluging for S2 position reconstruction at event level."""

    __version__ = "0.0.0"
    depends_on = ("event_area_per_channel", "event_basics")

    algorithm: Optional[str] = None
    compressor = "zstd"
    parallel = True  # can set to "process" after #82

    min_reconstruction_area = straxen.URLConfig(
        help="Skip reconstruction if area (PE) is less than this",
        default=0,
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
                "event_s2_x_" + self.algorithm,
                np.float32,
                f"Reconstructed {self.algorithm} S2 X position (cm), uncorrected",
            ),
            (
                "event_s2_y_" + self.algorithm,
                np.float32,
                f"Reconstructed {self.algorithm} S2 Y position (cm), uncorrected",
            ),
            (
                "event_alt_s2_x_" + self.algorithm,
                np.float32,
                f"Reconstructed {self.algorithm} alt S2 X position (cm), uncorrected",
            ),
            (
                "event_alt_s2_y_" + self.algorithm,
                np.float32,
                f"Reconstructed {self.algorithm} alt S2 Y position (cm), uncorrected",
            ),
        ]
        dtype += strax.time_fields
        return dtype

    def get_tf_model(self):
        """Simple wrapper to have several tf_event_model_mlp, tf_event_model_cnn, ..

        point to this same function in the compute method

        """
        model = getattr(self, f"tf_event_model_{self.algorithm}", None)
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

    def compute(self, events):
        result = np.ones(len(events), dtype=self.dtype)
        result["time"], result["endtime"] = events["time"], strax.endtime(events)
        result["event_s2_x_" + self.algorithm] *= float("nan")
        result["event_s2_y_" + self.algorithm] *= float("nan")
        result["event_alt_s2_x_" + self.algorithm] *= float("nan")
        result["event_alt_s2_y_" + self.algorithm] *= float("nan")

        model = self.get_tf_model()

        for p_type in ["s2", "alt_s2"]:
            peak_mask = events[p_type + "_area"] > self.min_reconstruction_area
            if not np.sum(peak_mask):
                continue

            _top_pattern = events[p_type + "_area_per_channel"][peak_mask, 0 : self.n_top_pmts]
            with np.errstate(divide="ignore", invalid="ignore"):
                _top_pattern = _top_pattern / np.max(_top_pattern, axis=1).reshape(-1, 1)

            _top_pattern = _top_pattern.reshape(-1, self.n_top_pmts)
            _pos = model.predict(_top_pattern)

            result["event_" + p_type + "_x_" + self.algorithm][peak_mask] = _pos[:, 0]
            result["event_" + p_type + "_y_" + self.algorithm][peak_mask] = _pos[:, 1]

        return result
