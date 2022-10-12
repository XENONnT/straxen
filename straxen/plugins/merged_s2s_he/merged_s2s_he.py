import strax
import straxen

import numpy as np
import numba
import pandas as pd

import typing as ty
from immutabledict import immutabledict
from straxen.plugins.defaults import HE_PREAMBLE
from straxen.plugins.merged_s2s.merged_s2s import MergedS2s

export, __all__ = strax.exporter()


@export
class MergedS2sHighEnergy(MergedS2s):
    __doc__ = HE_PREAMBLE + MergedS2s.__doc__
    depends_on = ('peaklets_he', 'peaklet_classification_he')
    data_kind = 'merged_s2s_he'
    provides = 'merged_s2s_he'
    __version__ = '0.0.1'
    child_plugin = True

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps['peaklets_he'].dtype_for('peaklets_he'))

    def compute(self, peaklets_he):
        # There are not any lone hits for the high energy channel,
        #  so create a dummy for the compute method.
        lone_hits = np.zeros(0, dtype=strax.hit_dtype)
        return super().compute(peaklets_he, lone_hits)
