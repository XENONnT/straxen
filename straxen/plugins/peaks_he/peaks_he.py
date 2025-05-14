import numpy as np
import strax
from straxen.plugins.defaults import HE_PREAMBLE
from straxen.plugins.peaks.peaks_vanilla import PeaksVanilla

export, __all__ = strax.exporter()


@export
class PeaksHighEnergy(PeaksVanilla):
    __doc__ = HE_PREAMBLE + (PeaksVanilla.__doc__ or "")

    __version__ = "0.0.1"
    depends_on = ("peaklets_he", "peaklet_classification_he", "merged_s2s_he")
    data_kind = "peaks_he"
    provides = "peaks_he"
    child_ends_with = "_he"

    def infer_dtype(self):
        return self.deps["peaklets_he"].dtype_for("peaklets")

    def compute(self, peaklets_he, merged_s2s_he):
        indicator_dtype = self.deps["merged_s2s_he"].indicator_dtype
        _peaklets_he = strax.merge_arrs(
            [peaklets_he, np.zeros(len(peaklets_he), dtype=indicator_dtype)],
            dtype=strax.merged_dtype((peaklets_he.dtype, indicator_dtype)),
        )
        return super().compute(_peaklets_he, merged_s2s_he)
