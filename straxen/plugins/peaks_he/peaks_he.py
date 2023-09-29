import strax
from straxen.plugins.defaults import HE_PREAMBLE
from straxen.plugins.peaks.peaks import Peaks

export, __all__ = strax.exporter()


@export
class PeaksHighEnergy(Peaks):
    __doc__ = HE_PREAMBLE + (Peaks.__doc__ or "")
    depends_on = ("peaklets_he", "peaklet_classification_he", "merged_s2s_he")
    data_kind = "peaks_he"
    provides = "peaks_he"
    __version__ = "0.0.1"
    child_ends_with = "_he"

    def infer_dtype(self):
        return self.deps["peaklets_he"].dtype_for("peaklets")

    def compute(self, peaklets_he, merged_s2s_he):
        return super().compute(peaklets_he, merged_s2s_he)
