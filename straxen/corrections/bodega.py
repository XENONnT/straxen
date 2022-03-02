

import strax
import rframe
import datetime

from .base_corrections import BaseCorrectionSchema

export, __all__ = strax.exporter()


@export
class Bodega(BaseCorrectionSchema):
    '''Detector parameters'''
    _NAME = 'bodega'
    
    field: str = rframe.Index()

    value: float
    uncertainty: float
    definition: str
    reference: str
    date: datetime.datetime

