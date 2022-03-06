

import strax
import rframe
import datetime
from typing import Literal

from .base_references import ResourceReference

export, __all__ = strax.exporter()


class BaseMap(ResourceReference):
    
    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    value: str


@export
class S1XYZMap(BaseMap):
    _NAME = "s1_xyz_maps"    


@export
class S2XYMap(BaseMap):
    _NAME = "s2_xy_maps"


@export
class FdcMap(BaseMap):
    _NAME = "fdc_maps"
    fmt = 'json.gz'
