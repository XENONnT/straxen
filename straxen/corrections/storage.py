import pymongo
from collections import defaultdict



class InsertionError(Exception):
    pass
class BaseCorrectionStore:
    
    def insert(self, name, doc):
        return self._insert(name, doc)
    
    def _insert(self, name, doc):
        raise NotImplementedError
    
    def update(self, name, index, doc):
        return self._update(name, index, doc)

    def _update(self, name, index, doc):
        raise NotImplementedError
        
    def find(self, name, lt={}, gt={}, eq={}):
        raise NotImplementedError
    
    def find(self, name, **kwargs):
        raise NotImplementedError
    
    def find_one(self, name, **kwargs):
        raise NotImplementedError
        
    def find_after(self, name, key, value,
                   limit=1,
                   inclusive=False,
                   **query):
        raise NotImplementedError
    
    def find_before(self, name, key, value, 
                    limit=1,
                    inclusive=True,
                    **query):
        raise NotImplementedError 
        
    def overlaps(self, name, begin, end=None,
                     begin_name='begin',
                     end_name='end',
                     closed_begin=True,
                     closed_end=False,
                     **query):
        raise NotImplementedError
    
class MongoCorrectionStore(BaseCorrectionStore):
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
    
    def find(self, name, **kwargs):
        return list(self.db[name].find(kwargs))

    def find_one(self, name, sort=None, **kwargs):
        return self.db[name].find_one(kwargs, sort=sort)
    
    def _insert(self, name, doc):
        return self.db[name].insert_one(doc)
    
    def _update(self, name, index, doc):
        self.db[name].update_one(index, doc, upsert=True)

    def find_after(self, name, key, value,
                   limit=1,
                   inclusive=False,
                   **query):
        collection = self.db[name]
        op  = "$gt"
        if inclusive:
            op += 'e'
        query[key] = {op: value}
        c = collection.find(query)
        c = c.sort(key, pymongo.ASCENDING)
        c = c.limit(limit)
        return list(c)
    
    def find_before(self, name, key, value, 
                    limit=1,
                    inclusive=True,
                    **query):
        collection = self.db[name]
        op  = "$lt"
        if inclusive:
            op += 'e'
        query[key] = {op: value}
        c = collection.find(query)
        c = c.sort(key, pymongo.DESCENDING)
        c = c.limit(limit)
        return list(c)
    
    def overlaps(self, name, begin, end=None,
                     begin_name='begin',
                     end_name='end',
                     closed_begin=True,
                     closed_end=False,
                     **query):
        query = defaultdict(dict, **query)
        begin_op = '$gte' if closed_end else '$gt'
        query[end_name][begin_op] = begin
        if end is not None:
            end_op = '$lte' if closed_begin else '$lt'
            query[begin_name][end_op] = end
        projection = {'_id': 0}
        return list(self.db[name].find(query, 
                                    projection=projection))

    def unique_values(self, name, field, **filter):
        return self.db[name].distinct(field, filter)