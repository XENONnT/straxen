import re
import pytz
import strax
import utilix
import pandas as pd

from pydantic import BaseModel
from typing import ClassVar, Union

from .indexers import Indexer


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
    name: ClassVar = 'correction'
    version: ClassVar = Indexer()
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
    