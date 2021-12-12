
import strax
import pandas as pd

from typing import Any, Type
from .correction import BaseCorrection

export, __all__ = strax.exporter()

@export
class CorrectionClient:
    correction: Type[BaseCorrection]
    db: Any
    
    def __init__(self, correction, db):
        self.correction = correction
        self.db = db
        
    def get(self, *args, **kwargs):
        docs = self.correction.index.query_db(self.db, *args, **kwargs)
        return docs
    
    def get_df(self, *args, **kwargs):
        docs = self.get(*args, **kwargs)
        df = pd.DataFrame(docs)
        idx = [c for c in self.correction.index.query_fields if c in df.columns]
        return df.sort_values(idx).set_index(idx)

    def set(self, *args, **kwargs):
        doc = self.correction(**kwargs)
        return self.correction.index.set(self.db, doc)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        docs = self.get(*index)
        nfields = len(self.correction.index.query_fields)
        if len(index)>nfields:
            for k in index[nfields:]:
                docs = [doc[k] for doc in docs]
        if len(docs)==1:
            return docs[0]
        return docs

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        if not isinstance(value, dict):
            value = {'value': value}
        self.set(*key, **value)

@export
class CorrectionsClient:
    db: Any

    def __init__(self, db):
        self.db = db
    
    @classmethod
    def default(cls):
        import pymongo
        db = pymongo.MongoClient()['cmt2']
        return cls(db)

    @property
    def corrections(self):
        return BaseCorrection.correction_classes()
    
    @property
    def correction_names(self):
        return list(self.corrections)

    def client(self, name):
        correction = self.corrections[name]
        return CorrectionClient(correction, self.db)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key[0] in self.corrections:
            return self.client(key[0])[key[1:]]
        
        if key in self.corrections:
            return self.client(key)
        raise KeyError(key)
        
    def __dir__(self):
        return super().__dir__() + list(self.corrections)
    
    def __getattr__(self, name):
        if name in self.corrections:
            return self[name]
        raise AttributeError(name)

    def get(self, correction_name, *args, **kwargs):
        return self.client(correction_name).get(*args, **kwargs)

    def set(self, correction_name, **kwargs):
        return self.client(correction_name).set(**kwargs)
