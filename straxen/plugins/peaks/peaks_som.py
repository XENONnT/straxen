import strax
import numpy as np
from straxen.plugins.peaks.peaks_vanilla import PeaksVanilla

export, __all__ = strax.exporter()


@export
class PeaksSOM(PeaksVanilla):
    """Same as Peaks but include in addition SOM type field to be propagated to event_basics.

    Thus, only change dtype.

    """

    __version__ = "0.0.1"
    child_plugin = True

    def infer_dtype(self):
        peaklet_classification_dtype = self.deps["enhanced_peaklet_classification"].dtype_for(
            "enhanced_peaklet_classification"
        )
        peaklets_dtype = self.deps["peaklets"].dtype_for("peaklets")
        # The merged dtype is argument position dependent!
        # It must be first classification then peaklet
        # Otherwise strax will raise an error when checking for the returned dtype!
        merged_dtype = strax.merged_dtype((peaklet_classification_dtype, peaklets_dtype))
        return merged_dtype

    def compute(self, peaklets, merged_s2s):
        result = super().compute(peaklets, merged_s2s)

        # For merged_s2s SOM and straxen type are undefined:
        # have to consider the peaks overlapping
        _is_merged_s2 = np.isin(result["time"], merged_s2s["time"]) & np.isin(
            strax.endtime(result), strax.endtime(merged_s2s)
        )
        result["vanilla_type"][_is_merged_s2] = -1
        result["som_sub_type"][_is_merged_s2] = -1

        return result
