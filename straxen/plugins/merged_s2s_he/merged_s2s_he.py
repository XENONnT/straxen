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

    @property
    def n_tpc_pmts(self):
        # Have to hack the url config to avoid nasty numba errors for the main MergedS2s plugin
        return self.n_he_pmts

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps["peaklets_he"].dtype_for("peaklets_he"))

    def compute(self, peaklets_he, start, end):
        # There are not any lone hits for the high energy channel,
        # so create a dummy for the compute method.
        lone_hits = np.zeros(0, dtype=strax.hit_dtype)
        return super().compute(peaklets_he, lone_hits, start, end)
