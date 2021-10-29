import pymongo
import pandas as pd

from .correction import BaseCorrection

class BaseCorrectionStore:
    class_ = BaseCorrection
    
    def insert(self, **kwargs):
        correction = self.class_(**kwargs)
        return self._insert(correction)
    
    def _insert(self, correction):
        raise NotImplementedError
    
    def find(self, **kwargs):
        raise NotImplementedError
    
    def find_one(self, **kwargs):
        raise NotImplementedError
        
    def get_values(self, **kwargs):
        return [c.get_value() for c in self.find(**kwargs)]
    
    def get_value(self, **kwargs):
        return self.find_one().get_value()
    
class PandasCorrectionStore(BaseCorrectionStore):
    df: pd.DataFrame
    
    def __init__(self, df=None):
        if df is None:
            df = pd.DataFrame(columns=list(self.class_.schema()['properties']))
        self.df = df
        
    def _insert(self, correction):
        row = correction.dict()
        self.df = self.df.append(row, ignore_index=True)
        
    def find(self, **kwargs):
        df = self.df.copy()
        for k,v in kwargs.items():
            df = df[df[k]==v]
        df = df.sort_values('version')
        return [self.class_(**d) for d in df.to_dict(orient='records')]
    
    def find_one(self, **kwargs):
        return self.find(**kwargs)[-1]
    
class MongoCorrectionStore(BaseCorrectionStore):
    _db: pymongo.MongoClient = None
    dbname: str
    collection: str
    
    def __init__(self, dbname, collection, **connection_kwargs):            
        self.dbname = dbname
        self.collection = collection
        self.connection_kwargs = connection_kwargs    
    
    @property
    def db(self):
        if self._db is None:
            self._db = pymongo.MongoClient(**self.connection_kwargs)[self.dbname]
        return self._db
    
    def find(self, **kwargs):
        return [self.class_(**d) for d in self.db[self.collection].find(kwargs)]
    
    def find_one(self, **kwargs):
        d = self.db[self.collection].find_one(**kwargs, sort=[('version', pymongo.DESCENDING)])
        return self.class_(**d)
    

    def _insert(self, correction):
        if not isinstance(correction, self.class_):
            correction = self.class_(**correction)
        return self.db[self.collection].insert_one(correction.dict())
    