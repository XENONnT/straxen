
import pymongo
import strax
import pandas as pd

from typing import Any, Type, Union
from .schema import BaseSchema

export, __all__ = strax.exporter()

@export
class RemoteDataframe:
    schema: Type[BaseSchema]
    db: Any
    
    def __init__(self, schema, db):
        if isinstance(db, str) and db.startswith('mongodb'):
            db = pymongo.MongoClient(db)
        if isinstance(db, str) and db.endswith('.csv'):
            db = pd.read_csv(db)
        if isinstance(db, str) and db.endswith('.pkl'):
            db = pd.read_pickle(db)
        if isinstance(db, str) and db.endswith('.pq'):
            db = pd.read_parquet(db)
        self.schema = schema
        self.db = db
    
    @property
    def columns(self):
        return self.schema.columns()

    @property
    def loc(self):
        return LocIndexer(self)

    @property
    def at(self):
        return AtIndexer(self)

    def get(self, *args, **kwargs):
        docs = self.schema.index.query_db(self.db, *args, **kwargs)
        return docs
            
    def get_df(self, *args, **kwargs):
        docs = self.get(*args, **kwargs)
        df = pd.DataFrame(docs, columns=list(self.schema.all_fields()))
        idx = [c for c in self.schema.index.query_fields if c in df.columns]
        return df.sort_values(idx).set_index(idx)

    def set(self, *args, **kwargs):
        doc = self.schema(**kwargs)
        return doc.save(self.db, *args, **kwargs)

    def __getitem__(self, index):
        if isinstance(index, str) and index in self.columns:
            return RemoteSeries(self, index)
        if isinstance(index, tuple) and index[0] in self.columns:
            return RemoteSeries(self, index[0])[index[1:]]
        raise KeyError(f'{index} is not a dataframe column.')

    def __dir__(self):
        return self.columns + super().__dir__()

    def __getattr__(self, name):
        if name in self.columns:
            return self[name]
        raise AttributeError(name)


class RemoteSeries:
    obj: RemoteDataframe
    column: str

    def __init__(self, obj, column):
        self.obj = obj
        self.column = column

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        return self.obj.get_df(*index)[self.column]

class Indexer:
    def __init__(self, obj: RemoteDataframe):
        self.obj = obj

class LocIndexer(Indexer):

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        index_fields = self.obj.schema.index_names()
        nfields = len(index_fields)
        if len(index)>nfields+1:
            raise 
        df = self.obj.get_df(*index)
    
        if len(index)==nfields+1:
            columns = index[-1]
            if not isinstance(columns, list):
                columns = [columns]
            df = df[columns]
        return df

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        if not isinstance(value, dict):
            value = {'value': value}
        self.obj.set(*key, **value)

class AtIndexer(Indexer):

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.obj.schema.index_names()) + 1:
            raise KeyError('Under defined index.')
        docs = self.obj.get(*index[:-1])
        docs = [doc[index[-1]] for doc in docs]
        if len(docs)==1:
            return docs[0]
        return docs
