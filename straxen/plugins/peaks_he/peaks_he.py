import numba
import numpy as np
import strax
from immutabledict import immutabledict
from strax.processing.general import _touching_windows
import straxen
from straxen.plugins.defaults import HE_PREAMBLE
from straxen.plugins.peaks.peaks import Peaks
from straxen.plugins.defaults import FAKE_MERGED_S2_TYPE

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
