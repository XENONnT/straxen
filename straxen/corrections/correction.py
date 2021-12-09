import re
import pytz
import time
import strax
import utilix
import pandas as pd
import datetime
from pydantic import BaseModel
from typing import ClassVar, Union

from .indexers import Indexer, InterpolatedIndexer, IntervalIndexer


export, __all__ = strax.exporter()

# editing will not be allowed for for time periods
# this many seconds into the future 
EDITING_BUFFER = 12*3600 

def camel_to_snake(name):
  name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

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
    name: ClassVar = ''
    value: Union[str,int,float]
    
    @classmethod
    def indices(cls):
        indices = {}
        for base in reversed(cls.mro()):
            if issubclass(base, BaseCorrection) and '_indices' in vars(base):
                indices.update(base._indices)
        return indices
    
    @classmethod
    def index_fields(cls):
        fields = set()
        for index in cls.indices().values():
            fields.update(index.fields)
        return fields
    
    @classmethod
    def all_fields(cls):
        fields = cls.index_fields()
        fields.update(cls.schema()['properties'])
        return fields

    @classmethod
    def correction_classes(cls):
        return {c.name: c for c in all_subclasses(cls) if c.name}

    def pre_insert(self, **index):
        pass

    def pre_update(self, old, **index):
        pass
    

@export
class TimeIntervalCorrection(BaseCorrection):
    version: ClassVar = Indexer(type=int)
    time: ClassVar = IntervalIndexer(type=datetime.datetime, left_name='begin', right_name='end')
    
    def pre_insert(self, **index):
        begin = pd.to_datetime(index['begin'], utc=True)
        cutoff = pd.to_datetime(time.time()+3600, unit='s', utc=True)
        if index['version']==0 and begin<cutoff:
            raise ValueError(f'Can only insert online intervals begining at least two hours in the future.')

def can_extrapolate(index):
    if index.get('version', 1):
        return False
    now = pd.to_datetime(time.time(), unit='s', utc=True)
    ts = pd.to_datetime(index.get('time', now), utc=True)
    return ts < now
        
@export
class TimeSampledCorrection(BaseCorrection):
    version: ClassVar = Indexer(type=int)
    time: ClassVar = InterpolatedIndexer(extrapolate=can_extrapolate)
        
    def pre_insert(self, **index):
        cutoff = pd.to_datetime(time.time()+3600, unit='s', utc=True)
        ts = pd.to_datetime(index['time'], utc=True)
        if index['version']==0 and ts<cutoff:
            raise ValueError(f'Can only insert online values for times at least two hours in the future.')
    