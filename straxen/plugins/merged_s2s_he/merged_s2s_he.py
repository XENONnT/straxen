import strax
import numpy as np

from straxen.plugins.defaults import HE_PREAMBLE
from straxen.plugins.merged_s2s.merged_s2s import MergedS2s

export, __all__ = strax.exporter()


@export
class MergedS2sHighEnergy(MergedS2s):
    __doc__ = HE_PREAMBLE + (MergedS2s.__doc__ or "")
    depends_on = ("peaklets_he", "peaklet_classification_he")
    data_kind = "merged_s2s_he"
    provides = "merged_s2s_he"
    __version__ = "0.1.0"
    child_plugin = True

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps["peaklets_he"].dtype_for("peaklets_he"))

    @property
    def use_bayesian_merging(self):
        # there is no pos-rec for HE channels
        return False

    def compute(self, peaklets_he, start, end):
        if self.no_merging(peaklets_he):
            return self.empty_result()

        _peaklets_he = strax.merge_arrs(
            [peaklets_he, np.zeros(len(peaklets_he), dtype=self.indicator_dtype)],
            dtype=strax.merged_dtype((peaklets_he.dtype, self.indicator_dtype)),
        )

        # There are not any lone hits for the high energy channel,
        # so create a dummy for the compute method.
        lone_hits = np.zeros(0, dtype=strax.hit_dtype)
        return self.merge(_peaklets_he, lone_hits, start, end)[0]
