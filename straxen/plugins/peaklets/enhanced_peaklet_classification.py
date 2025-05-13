from typing import Tuple
import numpy as np
import strax
import straxen
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO, UNCERTAIN_XYPOS_S2_TYPE


class EnhancedPeakletClassification(strax.Plugin):
    """Classify peaklets based on additional features and criteria."""

    __version__ = "0.0.0"
    depends_on: Tuple[str, ...] = (
        "peaklets",
        f"peaklet_positions_{DEFAULT_POSREC_ALGO}",
        "merged_peaklet_classification",
    )
    provides = "enhanced_peaklet_classification"
    data_kind = "peaklets"

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )

    cnf_contour_area_coeff = straxen.URLConfig(
        default=[-0.005, 0.219, -3.467, 15.610],
        type=(list, tuple),
        help="Coefficient of CNF contour area cut",
    )

    def infer_dtype(self):
        return self.deps["merged_peaklet_classification"].dtype_for("merged_peaklet_classification")

    @staticmethod
    def apply(peaklets, coefficients):
        mask = peaklets["position_contour_area_cnf"] < np.exp(
            np.polyval(
                coefficients,
                np.log(peaklets["area"]),
            )
        )
        # only apply the selection on type 2, after the classification during S2 merging
        mask |= peaklets["type"] != 2
        return mask

    def compute(self, peaklets):
        name = f"position_contour_{self.default_reconstruction_algorithm}"
        if name not in peaklets.dtype.names:
            raise ValueError(f"{name} is not in the input peaklets dtype")

        results = strax.merge_arrs([peaklets], dtype=self.dtype, replacing=True)
        mask = self.apply(peaklets, self.cnf_contour_area_coeff)
        results[results["type"] == 2] = np.where(
            mask[results["type"] == 2], 2, UNCERTAIN_XYPOS_S2_TYPE
        )

        return results
