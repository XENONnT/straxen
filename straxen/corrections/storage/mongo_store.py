

from dask.utils import Dispatch
import pymongo
from itertools import product
from .store import CorrectionStore
from ..indexers import Indexer, InterpolatedIndexer, IntervalIndexer

class MongoCorrectionStore(CorrectionStore):
    index_query = Dispatch('index_query')
    _db: pymongo.MongoClient = None
    dbname: str

    def __init__(self, dbname, **connection_kwargs):            
        self.dbname = dbname
        self.connection_kwargs = connection_kwargs    

    def __reduce__(self):
        return (MongoCorrectionStore,
                (self.dbname, ),
                self.connection_kwargs)
    @property
    def db(self):
        if self._db is None:
            client = pymongo.MongoClient(**self.connection_kwargs)
            self._db = client[self.dbname]
        return self._db
    
    def build_mongo_queries(self, correction_indices, index):
        queries = []
        
        for k,v in index.items():
            if k in correction_indices:
                sub_queries = self.index_query(correction_indices[k], v)
            else:
                sub_queries = [dict(filter={k: v})]
            queries.append(sub_queries)

        for sub_queries in product(*queries):
            query = {'filter': {}, 'sort': []}
            for sub_query in sub_queries:
                if 'filter' in sub_query:
                    query['filter'].update(sub_query['filter'])
                if 'sort' in sub_query:
                    query['sort'].append(sub_query['sort'])
                if 'limit' in sub_query:
                    query['limit'] = sub_query['limit']
            yield query
    
    def get_values(self, correction, *args, **kwargs):
        if not isinstance(correction, type):
            correction = correction.__class__
        index = self.construct_index(correction, *args, **kwargs)
        records = []
        for query in self.build_mongo_queries(correction.indices(), index):
            for d in self.db[correction.name].find(projection={'_id': 0}, **query):
                record = correction(**d).dict()
                for k,v in correction.indices().items():
                    record[k] = v.construct_index(d)
                records.append(record)
                
        for indx in correction.indices().values():
            records = indx.process_records(records, index)
        return records
    
    def get_value(self, correction, *args, **kwargs):
        index = self.construct_index(correction, *args, **kwargs)
        index.update(kwargs)
        if set(index).symmetric_difference(correction.indices()):
            raise ValueError(f'get_value method only supports exact index lookup.\
            A value for all of the indices: {list(correction.indices())} must be provided')
        values = self.get_values(correction, **index)
    
@MongoCorrectionStore.index_query.register(Indexer)
def index_query(index, value):
    return [dict(filter={index.name: value})]

@MongoCorrectionStore.index_query.register(IntervalIndexer)
def interval_index_query(index, value):
    if isinstance(value, tuple) and len(value)==2:
        left, right = value
    elif isinstance(value, slice):
        left, right = value.start, value.stop
    else:
        left = right = value
    if left>right:
        left, right = right, left
    query = {}
    right_op = '$gte' if index.closed in ['right', 'both'] else '$gt'
    query[index.right_name] = {right_op: left}
    left_op = '$lte' if index.closed in ['left', 'both'] else '$lt'
    query[index.left_name] = {left_op: right}
    return [dict(filter=query)]

@MongoCorrectionStore.index_query.register(InterpolatedIndexer)
def interpolated_index_query(index, value):
    queries = []
    after_query = {}
    op  = "$gt"
    if index.inclusive:
        op += 'e'
    after_query[index.name] = {op: value}
    query = dict(filter=after_query, sort=(index.name, pymongo.ASCENDING), limit=1)
    queries.append(query)
    
    before_query = {}
    op  = "$lt"
    if index.inclusive:
        op += 'e'
    before_query[index.name] = {op: value}
    query = dict(filter=before_query, sort=(index.name, pymongo.DESCENDING), limit=1)
    queries.append(query)
    return queries