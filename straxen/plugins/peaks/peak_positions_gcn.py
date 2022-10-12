import strax
import straxen

import numpy as np
import numba
import pandas as pd

import typing as ty
from immutabledict import immutabledict
from straxen.plugins.peaks._peak_positions_base import PeakPositionsBaseNT

export, __all__ = strax.exporter()


@export
class PeakPositionsGCN(PeakPositionsBaseNT):
    """Graph Convolutional Network (GCN) neural net for position reconstruction"""
    provides = "peak_positions_gcn"
    algorithm = "gcn"
    __version__ = '0.0.1'

    tf_model_gcn = straxen.URLConfig(
        default=f'tf://'
                f'resource://'
                f'cmt://{algorithm}_model'
                f'?version=ONLINE'
                f'&run_id=plugin.run_id'
                f'&fmt=abs_path',
        help='GCN model. Should be opened using the "tf" descriptor. '
             'Set to "None" to skip computation',
        cache=3,
    )
