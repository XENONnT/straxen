
import strax
import rframe
import datetime
from typing import Literal
from .base_references import ResourceReference

export, __all__ = strax.exporter()


@export
class TFModel(ResourceReference):
    _NAME = "tf_models"
    fmt = 'json'

    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    value: str
