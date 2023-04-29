import strax
import straxen
from straxen.plugins.merged_s2s import MergedS2s

import numpy as np
import numba


export, __all__ = strax.exporter()


@export
class MergedS2sTiming(MergedS2s):
    """
    Merge together peaklets if peak finding favours that they would
    form a single peak instead.
    """
    __version__ = '0.0.0'

    depends_on = ('peaklet_timing', 'peaklet_classification', 'lone_hits')
    data_kind = 'merged_s2s_timing'
    provides = 'merged_s2s_timing'

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps['peaklet_timing'].dtype_for('peaklet_timing'))
