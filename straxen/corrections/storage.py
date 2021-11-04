import pymongo
import pandas as pd
import pytz
import utilix
import strax
from straxen.corrections.utils import TypeDispatch

from .model import BaseCorrectionModel, BaseIntervalCorrection


class InsertionError(Exception):
    pass

class BaseCorrectionStore:

    def insert(self, index, doc=None, **kwargs):
        if doc is None:
            doc = kwargs
        return self._insert(doc)
    
    def _insert(self, doc):
        raise NotImplementedError
    
    def find(self, index, **kwargs):
        raise NotImplementedError
    
    def find_one(self, index, **kwargs):
        raise NotImplementedError
        
    def get_values(self, index, **kwargs):
        return [c['value'] for c in self.find(**kwargs)]
    
    def get_value(self, index, **kwargs):
        doc = self.find_one( index, **kwargs)
        if doc:
            return doc['value']

    def update(self, index, doc=None, **kwargs):
        if doc is None:
            doc = kwargs
        return self._update(index, doc)

    def _update(self, index, doc):
        raise NotImplementedError

class BaseInterpolatingStore(BaseCorrectionStore):
    def interpolate(self, index, **kwargs):
        raise NotImplementedError
        
class BaseIntervalStore(BaseCorrectionStore):
    def overlaps(self, index, **kwargs):
        raise NotImplementedError
    
    def value_at(self, index, **kwargs):
        raise NotImplementedError

class DictCorrectionStore(BaseCorrectionStore):
    data: dict
    
    def __init__(self, data=None):
        if data is None:
            data = {}
        self.data = data
        
    def _insert(self, index, correction):
        token = strax.deterministic_hash(index)
        if token in self.data:
            raise InsertionError(f'A correction with index {index} already exists.')
        self.data[token] = correction
        
    def find(self, index,  **kwargs):
        token = strax.deterministic_hash(index)
        return self.data.get(token, None)
    
    def find_one(self, index,  **kwargs):
        return self.find(**kwargs)[-1]
    
class MongoCorrectionStore(BaseCorrectionStore):
    _db: pymongo.MongoClient = None
    dbname: str

    def __init__(self, dbname, collection, **connection_kwargs):            
        self.cname = collection
        self.dbname = dbname
        self.connection_kwargs = connection_kwargs    

    def __reduce__(self):
        return (MongoCorrectionStore,
                (self.model, self.dbname, self.cname),
                self.connection_kwargs)

    @property
    def db(self):
        if self._db is None:
            self._db = pymongo.MongoClient(**self.connection_kwargs)
        return self._db

    @property
    def collection(self):
        return self.db[self.dbname][self.cname]
        
    def find(self, index, **kwargs):
        kwargs.update(index)
        return list(self.collection.find(kwargs))

    def find_df(self, index, **kwargs):
        return pd.DataFrame(self.find(index, **kwargs))

    def find_one(self, index, **kwargs):
        kwargs.update(index)
        return self.collection.find_one(kwargs, sort=[('version', pymongo.DESCENDING)])
    
    def _insert(self, index, data):
        if self.find_one(index):
            raise InsertionError(f'A correction with index {index} already exists.')
        return self.collection.insert_one(data)


class MongoIntervalStore(BaseIntervalStore, MongoCorrectionStore):
    
    def overlaps(self, begin, end=None, **kwargs):
        if isinstance(begin, slice):
            end = begin.stop
            begin = begin.start
        elif isinstance(begin, pd.Interval):
            end = begin.right
            begin = begin.left
        elif isinstance(begin, tuple) and len(begin)==2:
            begin,end = begin
        query = kwargs
        query['$or'] = [{'end': None}, {'end': {'$gt': begin}}]
        if end is None:
            end = begin
        query['begin'] = {'$lte': end}
        return list(self.collection.find(query))

    def overlaps_df(self, begin, end=None, **kwargs):
        return pd.DataFrame(self.overlaps(begin, end=end, **kwargs))
    
    def _insert(self, index, data):
        begin = data.pop('begin')
        end = data.pop('end', None)
        index = {k:v for k,v in index.items() if k not in ['begin', 'end']}
        if self.overlaps(begin, end, **index):
            raise InsertionError('This correction overlaps with existing corrections.')
        return super()._insert(data)
    
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
        return pd.to_datetime(kwargs['time'])
    if 'run_id' in kwargs:
        return run_id_to_time(kwargs['run_id'])
    else:
        return None

