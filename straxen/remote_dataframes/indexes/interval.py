import strax
import pandas as pd

from .index import *
from ..utils import singledispatchmethod


export, __all__ = strax.exporter()

@export
class IntervalIndexMixin:

    left_name: str = 'left'
    right_name: str = 'right'
    closed: str = 'left'
        
    def __init__(self, left_name=None,
                 right_name=None, 
                 closed=None, **kwargs):
        super().__init__(**kwargs)
        if left_name is not None:
            self.left_name = left_name
        if right_name is not None:
            self.right_name = right_name
        if closed is not None:
            self.closed = closed

    @property
    def store_fields(self):
        return (self.left_name, self.right_name)

    def to_interval(self, value):
        if isinstance(value, list):
            return [self.to_interval(val) for val in value]
        left = right = None
        if isinstance(value, tuple):
            left, right = value
        elif isinstance(value, slice):
            left, right = value.start, value.stop 
        elif isinstance(value, pd.Interval):
            left, right = value.left, value.right
        elif isinstance(value, dict):
            left = value.get(self.left_name, None)
            right = value.get(self.right_name, None)
        else:
            left = right = value
        return (left, right)

    def infer_index_value(self, **kwargs):
        '''For user conveniece this function extracts the
        left and right side of the query interval from various
        ways they may have passed them to the query function
        e.g tuple, slice,Interval, dict, single value or as explicit kwargs
        '''

        if self.name in kwargs:
            return self.to_interval(kwargs[self.name])
        else:
            return self.to_interval(kwargs)

    def index_to_storage_doc(self, index):
        left, right = index[self.name]
        return {
            self.name: index[self.name],
            self.left_name: left,
            self.right_name: right,
            }

    def reduce(self, docs, value):
        if value is None:
            return docs

        for doc in docs:
            doc[self.name] = self.infer_index_value(**doc)
        return docs

    @singledispatchmethod
    def build_query(self, db, value):
        raise TypeError(f"{type(db)} backend not supported.")

    def builds(self, **kwargs):
        from hypothesis import strategies as st
        strategy = super().builds(**kwargs)

        return st.tuples(strategy, strategy).filter(lambda x: x[0]<x[1])


@export
class TimeIntervalIndex(IntervalIndexMixin, DatetimeIndex):
    pass


@export
class IntegerIntervalIntex(IntervalIndexMixin, IntegerIndex):
    pass