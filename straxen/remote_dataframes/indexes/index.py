
import strax
import datetime
from pandas.core.algorithms import isin
import pymongo
import numbers
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Callable, Union, Type
from .coercers import COERCERS
from ..utils import singledispatchmethod

export, __all__ = strax.exporter()


@export
class BaseIndex:
    name: str = ''
    type: Type = int
    coerce: Callable = None

    def __init__(self, name=None, 
                schema=None, nullable=True,
                ):

        self.nullable = nullable
        if name is not None:
            self.name = name
        if schema is not None:
            self.schema = schema
            
    def __set_name__(self, owner, name):
        self.schema = owner
        if not self.name:
            self.name = name

    @property
    def query_fields(self):
        return (self.name, )
        
    @property
    def store_fields(self):
        return (self.name,)

    def infer_index_value(self, **kwargs):
        value = kwargs.get(self.name, None)
        value = self.validate(value)
        return value

    def infer_index(self, *args, **kwargs):
        index = dict(zip(self.query_fields, args))
        index.update(kwargs)
        return {self.name: self.infer_index_value(**index)}

    def index_to_storage_doc(self, index):
        return {self.name: index[self.name]}

    def validate(self, value):
        if isinstance(value, slice):
            start = self.validate(value.start)
            stop = self.validate(value.stop)
            step = self.validate(value.step)
            if start is None and stop is None:
                value = None
            else:
                return slice(start, stop, step)

        if value is None and self.nullable:
            return value

        if isinstance(value, list) and self.type is not list:
            return [self.validate(val) for val in value]

        if isinstance(value, tuple) and self.type is not tuple:
            return tuple(self.validate(val) for val in value)

        value = self.coerce(value)
        
        if not isinstance(value, self.type):
            raise TypeError(f'{self.name} must be of type {self.type}')

        return value

    def coerce(self, value):
        dtype = self.type 

        if not isinstance(dtype, tuple) and self.type is not tuple:
            dtype = (dtype, )

        if isinstance(value, dtype):
            return value

        for t in dtype:
            try:
                value = COERCERS[t](value)
                break
            except:
                pass
        return value

    def query_db(self, db, *args, **kwargs):
        value = self.infer_index(*args, **kwargs)[self.name]
        query = self.build_query(db, value)
        docs = self.apply_query(db, query)
        docs = self.reduce(docs, value)
        return docs

    def reduce(self, docs, value):
        return docs

    @singledispatchmethod
    def build_query(self, db, value):
        raise TypeError(f"{type(db)} backend not supported.")

    @singledispatchmethod
    def apply_query(self, db, query):
        raise TypeError(f"{type(db)} backend not supported.")

    @apply_query.register(list)
    def apply_list(self, db, query):
        return [self.apply_query(d, query) for d in db]

    @apply_query.register(dict)
    def apply_dict(self, db, query):
        return {k: self.apply_query(d, query) for k,d in db.items()}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name},type={self.type})"

    def example(self, start=0, stop=10, size=10):
        return pd.RangeIndex


@export
class StringIndex(BaseIndex):
    type = str


@export
class IntegerIndex(BaseIndex):
    type = int

@export
class FloatIndex(BaseIndex):
    type = float

@export
class DatetimeIndex(BaseIndex):
    type = datetime.datetime
    utc: bool

    def __init__(self, utc=True, unit='s', **kwargs):
        super().__init__(**kwargs)
        self.utc = utc
        self.unit = unit

    def coerce(self, value):
        unit = self.unit if isinstance(value, numbers.Number) else None
        return pd.to_datetime(value, utc=self.utc, unit=unit).to_pydatetime()

@export
class TupleIndex(BaseIndex):
    type = tuple
