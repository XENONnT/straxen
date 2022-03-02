

import strax
import rframe
import datetime
from typing import Literal

from .base_references import ResourceReference

export, __all__ = strax.exporter()


@export
class S1XYZMap(ResourceReference):
    _NAME = "s1_xyz_maps"
    
    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    value: str


@export
class S2XYMap(ResourceReference):
    _NAME = "s2_xy_maps"

    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()
    
    value: str

@export
class FdcMapName(ResourceReference):
    _NAME = "fdc_map_names"
    fmt = 'json.gz'

    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()
    
    value: str
