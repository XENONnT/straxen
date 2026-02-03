import strax
import numpy as np
from straxen.plugins.peaklets.peaklet_classification_som import som_additional_fields
from straxen.plugins.peaks.peaks_vanilla import PeaksVanilla
from typing import Tuple, Union

export, __all__ = strax.exporter()


@export
class PeaksSOM(PeaksVanilla):
    """Same as Peaks but include in addition SOM type field to be propagated to event_basics.

    Thus, only change dtype.

    """

    __version__ = "0.0.1"
    child_plugin = True

    depends_on: Union[Tuple[str, ...], str] = (
        "peaklets",
        "enhanced_peaklet_classification",
        "merged_s2s",
    )

    def infer_dtype(self):
        # In case enhanced_peaklet_classification has more fields than peaklets,
        # we need to merge them
        peaklets_dtype = self.deps["peaklets"].dtype_for("peaklets")
        peaklet_classification_dtype = self.deps["enhanced_peaklet_classification"].dtype_for(
            "enhanced_peaklet_classification"
        )
        merged_dtype = strax.merged_dtype((peaklets_dtype, peaklet_classification_dtype))
        # Numba is very picky about alignment for structured dtypes. Ensure an aligned dtype
        # so assignments inside strax.replace_merged don't see "unaligned array(...)".
        return np.dtype(merged_dtype, align=True)

    def compute(self, peaklets, merged_s2s):
        som_additional = np.zeros(
            len(merged_s2s), dtype=strax.to_numpy_dtype(som_additional_fields)
        )
        strax.set_nan_defaults(som_additional)
        # make sure _merged_s2s and peaklets have same dtype
        _merged_s2s = strax.merge_arrs([merged_s2s, som_additional], dtype=peaklets.dtype)
        peaks = super().compute(peaklets, _merged_s2s)
        return peaks
