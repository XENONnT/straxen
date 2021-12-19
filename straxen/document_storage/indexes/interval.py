import pandas as pd
from pandas.core.algorithms import isin
from pandas.core.indexes import interval
import pymongo

from .index import Index
from ..utils import singledispatchmethod


class IntervalIndex(Index):
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

    def validate(self, value):
        if isinstance(value, list):
            return [self.validate(val) for val in value]
        
        if self.coerce is not None:
            value = self.coerce(value)
        if not isinstance(value, self.type):
            raise TypeError(f'{self.name} must be of type {self.type}')

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

    @singledispatchmethod
    def build_query(self, db, value):
        raise TypeError(f"{type(db)} backend not supported.")
