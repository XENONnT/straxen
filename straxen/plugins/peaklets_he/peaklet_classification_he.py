import strax
import straxen

import numpy as np
import numba
import pandas as pd

import typing as ty
from immutabledict import immutabledict

export, __all__ = strax.exporter()
import numba
import numpy as np
import strax
from immutabledict import immutabledict
from strax.processing.general import _touching_windows
import straxen
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO
from straxen.plugins.peaklets.peaklet_classification import PeakletClassification
from straxen.plugins.defaults import HE_PREAMBLE

@export
class PeakletClassificationHighEnergy(PeakletClassification):
    __doc__ = HE_PREAMBLE + PeakletClassification.__doc__
    provides = 'peaklet_classification_he'
    depends_on = ('peaklets_he',)
    __version__ = '0.0.2'
    child_plugin = True

    def compute(self, peaklets_he):
        return super().compute(peaklets_he)

