import strax
from straxen.plugins.peaks.peaks import Peaks

export, __all__ = strax.exporter()


@export
class PeaksSOM(Peaks):
    """Same as Peaks but include in addition SOM type field to be propagated to event_basics.

    Thus, only change dtype.

    """

    __version__ = "0.0.1"
    child_plugin = True

    def infer_dtype(self):
        peaklet_classification_dtype = self.deps["peaklet_classification"].dtype_for(
            "peaklet_classification"
        )
        peaklets_dtype = self.deps["peaklets"].dtype_for("peaklets")
        # The merged dtype is argument position dependent! It must be first classification then peaklet
        # Otherwise strax will raise an error when checking for the returned dtype!
        merged_s2s_dtype = strax.merged_dtype((peaklet_classification_dtype, peaklets_dtype))
        return merged_s2s_dtype

    def setup(self):
        self.copy_function_name = "_copy_requested_peak_fields_som"

    def compute(self, peaklets, merged_s2s):
        result = super().compute(peaklets, merged_s2s)
        return result
