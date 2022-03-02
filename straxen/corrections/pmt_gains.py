
import strax
import rframe
from typing import Literal

from .base_corrections import TimeSampledCorrection

export, __all__ = strax.exporter()


@export
class PmtGain(TimeSampledCorrection):
    _NAME = 'pmt_gains'
    
    # Here we use a simple indexer (matches on exact value)
    # to define the pmt field
    # this will add the field to all documents and enable
    # selections on the pmt number. Since this is a index field
    # versioning will be indepentent for each pmt
    
    detector: Literal['tpc', 'nveto','muveto'] = rframe.Index()
    pmt: int = rframe.Index()
    
    value: float
