import strax
import numpy as np
from straxen.plugins.peaklets.peaklet_classification_som import som_additional_fields
from straxen.plugins.peaks.peaks_vanilla import PeaksVanilla

export, __all__ = strax.exporter()


@export
class PeaksSOM(PeaksVanilla):
    """Same as Peaks but include in addition SOM type field to be propagated to event_basics.

    Thus, only change dtype.

    """

    __version__ = "0.0.1"
    child_plugin = True

    def compute(self, peaklets, merged_s2s):
        som_additional = np.zeros(
            len(merged_s2s), dtype=strax.to_numpy_dtype(som_additional_fields)
        )
        strax.set_nan_defaults(som_additional)
        # make sure _merged_s2s and peaklets have same dtype
        _merged_s2s = strax.merge_arrs([merged_s2s, som_additional], dtype=peaklets.dtype)
        peaks = super().compute(peaklets, _merged_s2s)
        return peaks
