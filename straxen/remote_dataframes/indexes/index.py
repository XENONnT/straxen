import pytz
import strax
import datetime
import numbers
import pandas as pd
from typing import Type
from ..utils import singledispatchmethod

export, __all__ = strax.exporter()


@export
class BaseIndex:
    name: str = ''
    type: Type

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
    def indexes(self):
        return [self]

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

        if isinstance(value, self.type):
            return value
        
        if hasattr(self, '_coerce'):
            return self._coerce(value)

    def query_db(self, db, *args, **kwargs):
        value = self.infer_index(*args, **kwargs)[self.name]
        query = self.build_query(db, value)
        docs = self.apply_query(db, query)
        docs = [dict(doc, **self.infer_index(**doc)) for doc in docs]
        docs = self.reduce(docs, value)
        return docs

    def reduce(self, docs, value):
        return docs

    @singledispatchmethod
    def head(self, db, n=10):
        raise TypeError('head is not supported'
                       f' for {type(db)} datastores')

    @singledispatchmethod
    def ensure_index(self, db):
        TypeError(f"Ensure index not supported on {type(db)} backend.")

    @singledispatchmethod
    def build_query(self, db, value):
        raise TypeError(f"build_query not supported on {type(db)} backend.")

    @singledispatchmethod
    def apply_query(self, db, query):
        raise TypeError(f"apply_query not supported on {type(db)} backend.")

    @apply_query.register(list)
    def apply_list(self, db, query):
        return [self.apply_query(d, query) for d in db]

    @apply_query.register(dict)
    def apply_dict(self, db, query):
        return {k: self.apply_query(d, query) for k,d in db.items()}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name},type={self.type})"


@export
class StringIndex(BaseIndex):
    type = str

    def _coerce(self, value):
        return str(value)

    def builds(self, **kwargs):
        from hypothesis import strategies as st
        from string import printable
        return st.text(printable, min_size=1)


@export
class IntegerIndex(BaseIndex):
    type = int
    posdef: bool

    def __init__(self, posdef=True, **kwargs):
        super().__init__(**kwargs)
        self.posdef = posdef 

    def _coerce(self, value):
        value = int(value)

        if self.posdef:
            value = max(0, value)
        return value

    def builds(self, max_value=1000, **kwargs):
        from hypothesis import strategies as st
        min_value = 0 if self.posdef else None
        min_value = kwargs.get('min_value', min_value)
        return st.integers(min_value=min_value, max_value=max_value)

@export
class FloatIndex(BaseIndex):
    type = float

    def _coerce(self, value):
        return float(value)

    def builds(self, **kwargs):
        from hypothesis import strategies as st
        return st.floats(**kwargs)

@export
class DatetimeIndex(BaseIndex):
    type = datetime.datetime
    utc: bool

    def __init__(self, utc=True, unit='s', **kwargs):
        super().__init__(**kwargs)
        self.utc = utc
        self.unit = unit

    def coerce(self, value):
        if isinstance(value, datetime.datetime):
            if value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None:
                value = value.astimezone(pytz.utc)
            else:
                value = value.replace(tzinfo=pytz.utc)
            return value
        unit = self.unit if isinstance(value, numbers.Number) else None
        value = pd.to_datetime(value, utc=self.utc, unit=unit).to_pydatetime()
        return self.coerce(value)

    def builds(self, **kwargs):
        from hypothesis import strategies as st
        from hypothesis.extra.pytz import timezones

        return st.datetimes(min_value=kwargs.get('min_value', pd.Timestamp.min+datetime.timedelta(days=365)),
                            max_value=kwargs.get('max_value', pd.Timestamp.max-datetime.timedelta(days=365)),
                            timezones=timezones())
