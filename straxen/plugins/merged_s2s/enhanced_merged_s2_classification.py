from straxen.plugins.defaults import DEFAULT_POSREC_ALGO
from straxen.plugins.peaklets.enhanced_peaklet_classification import EnhancedPeakletClassification


class EnhancedMergedS2Classification(EnhancedPeakletClassification):
    """Classify merged S2s based on additional features and criteria."""

    __version__ = "0.0.0"
    child_plugin = True
    depends_on = ("merged_s2s", f"merged_s2_positions_{DEFAULT_POSREC_ALGO}")
    provides = "enhanced_merged_s2_classification"
    data_kind = "merged_s2s"

    def infer_dtype(self):
        return self.deps["merged_s2s"].dtype_for("merged_peaklet_classification")

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
