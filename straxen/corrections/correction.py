import re
import pytz
import strax
import numpy as np
import pandas as pd

import typing as ty
from datetime import datetime
import time
from .storage import InsertionError, BaseCorrectionStore
import utilix
from scipy.interpolate import interp1d
from pydantic import BaseModel, Field


export, __all__ = strax.exporter()

# editing will not be allowed for for time periods
# this many seconds into the future 
EDITING_BUFFER = 12*3600 

def camel_to_snake(name):
  name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def run_id_to_time(run_id):
    run_id = int(run_id)
    runsdb = utilix.rundb.xent_collection()
    rundoc = runsdb.find_one(
            {'number': run_id},
            {'start': 1})
    if rundoc is None:
        raise ValueError(f'run_id = {run_id} not found')
    time = rundoc['start']
    return time.replace(tzinfo=pytz.utc)

def infer_time(kwargs):
    if 'time' in kwargs:
        return pd.to_datetime(kwargs['time'], utc=True)
    if 'run_id' in kwargs:
        return run_id_to_time(kwargs['run_id'])
    else:
        return None

@export
class BaseCorrection(BaseModel):
    _corrections: ty.ClassVar = {}
    index_fields: ty.ClassVar = ('version',)
    version: int = 0
    value: ty.Union[int,float,str]

    def __init_subclass__(cls):
        cls.name = camel_to_snake(cls.__name__)
        if cls.name.startswith('base'):
            return
        correction = cls._corrections.setdefault(cls.name, cls)
        if correction is not cls:
            raise ValueError(f"A correction with the name \
                             {cls.name} already exists.")
    
    @property
    def index(self):
        return {field: getattr(self, field) 
                for field in self.index_fields}

    @property
    def cutoff(self):
        return pd.to_datetime(time.time()+EDITING_BUFFER, unit='s', utc=True)

    def pre_insert(self, store):
        pass

    @classmethod
    def insert(cls, store, docs):
        if isinstance(docs, (ty.Mapping, cls)):
            docs = [docs]
        if not isinstance(docs, ty.Iterable):
            raise TypeError(f'docs must be of type Mapping or {cls} or iterable of those types.')
        docs = [doc if isinstance(doc, cls) else cls(**doc) for doc in docs]
        for doc in docs:
            doc.save(store)
        return docs

    def save(self, store):
        self.pre_insert(store)
        store.insert(self.name, self.dict())

    @classmethod
    def from_store(cls, store, **index):
        raise NotImplementedError
    
    @classmethod
    def validity(cls, store, begin, end, version):
        raise NotImplementedError

@export
class BaseSampledCorrection(BaseCorrection):
    time: datetime

    @staticmethod
    def interpolate(left, right, time):
        x1,x2 = left.time.timestamp(), right.time.timestamp()
        y1,y2 = left.value, right.value
        func = interp1d([x1,x2], [y1,y2], fill_value=(y1,y2), bounds_error=False)
        return np.asscalar(func(time.timestamp()))

    @classmethod
    def from_store(cls, store, time, version=0):
        time = pd.to_datetime(time, utc=True)
        before = store.find_before(cls.name, 'time', time, version=version)
        left = cls(**before[0])
        after = store.find_after(cls.name,'time', time, version=version)
        if len(after):
            right = cls(**after[0])
            left.value = cls.interpolate(left, right, time)
            left.time = time
            return left
        elif left.version:
            raise ValueError('Invalid time')
        return left

    def pre_insert(self, store):
        existing = self.__class__.from_store(store, self.time, version=self.version)
        if existing and existing.value != self.value:
            raise InsertionError(f'Values for {self.time} already set to {existing.value}.')
            
    def pre_delete(self, store):
        if self.time<self.cutoff:
            raise InsertionError(f'You cannot delete values before {self.cutoff}.')

    def pre_update(self, store):
        self.pre_insert(store)
    

@export
class BaseMappingCorrection(BaseCorrection):
    index_fields = ('version', 'key')
    key: ty.Union[str,int]

@export
class BaseIntervalCorrection(BaseCorrection):
    begin: ty.Union[int,datetime]
    end: ty.Union[int,datetime]

    @property
    def interval(self):
        return pd.Interval(self.begin, self.end)

@export
class BaseIntIntervalCorrection(BaseIntervalCorrection):
    begin: int
    end: int
    
@export
class BaseTimeIntervalCorrection(BaseIntervalCorrection):
    begin: datetime
    end: datetime

    @classmethod
    def from_store(cls, store: BaseCorrectionStore, time, version=0):
        time = pd.to_datetime(time, utc=True)
        overlaps = store.overlaps(cls.name, time, time, version=version)
        if len(overlaps):
            return cls(**overlaps[0])

    def pre_insert(self, store):
        overlaps = store.overlaps(self.name, 
                                self.begin,
                                self.end,
                                version=self.version)
        if overlaps:
            raise InsertionError(f'Document overlaps with existing documents.')

    