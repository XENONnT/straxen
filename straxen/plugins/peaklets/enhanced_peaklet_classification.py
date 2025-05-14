from typing import Tuple
import numpy as np
import strax
import straxen
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO, UNCERTAIN_XYPOS_S2_TYPE


class EnhancedPeakletClassification(strax.Plugin):
    """Classify peaklets based on additional features and criteria."""

    __version__ = "0.0.1"
    depends_on: Tuple[str, ...] = (
        "peaklets",
        f"peaklet_positions_{DEFAULT_POSREC_ALGO}",
        "enhanced_peaklet_classification",
    )
    provides = "_enhanced_peaklet_classification"
    data_kind = "peaklets"

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )

    def infer_dtype(self):
        return self.deps["enhanced_peaklet_classification"].dtype_for("enhanced_peaklet_classification")

    @staticmethod
    def apply(peaklets):
        mask = (
            peaklets["width"][:, 5] / 1e3
            < 24.59 * np.exp(-1.35 * (np.log10(peaklets["area"]) - 1.99)) + 9.93
        )
        # only apply the selection on type 2, after the classification during S2 merging
        mask |= peaklets["area"] > 2e4
        mask |= peaklets["type"] != 2
        return mask

    def compute(self, peaklets):
        results = strax.merge_arrs([peaklets], dtype=self.dtype, replacing=True)
        mask = self.apply(peaklets)
        results["type"][results["type"] == 2] = np.where(
            mask[results["type"] == 2], 2, UNCERTAIN_XYPOS_S2_TYPE
        )

        return results
