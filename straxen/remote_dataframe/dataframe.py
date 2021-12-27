
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

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        if not isinstance(value, dict):
            value = {'value': value}
        self.set(*key, **value)

class RemoteSeries:
    df: RemoteDataframe
    column: str

    def __init__(self, df, column):
        self.df = df
        self.column = column

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        docs = self.df.get_df(*index)

        docs = [doc[self.column] for doc in docs]
        return docs

class Indexer:
    def __init__(self, df):
        self.df = df

class LocIndexer(Indexer):

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        index_fields = self.df.schema.index_names()
        nfields = len(index_fields)
        if len(index)>nfields+1:
            raise 
        df = self.df.get_df(*index)
    
        if len(index)==nfields+1:
            columns = index[-1]
            if not isinstance(columns, list):
                columns = [columns]
            df = df[columns]
        return df

class AtIndexer(Indexer):

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.df.schema.index_names()) + 1:
            raise KeyError('Under defined index.')
        docs = self.df.get(*index[:-1])
        docs = [doc[index[-1]] for doc in docs]
        if len(docs)==1:
            return docs[0]
        return docs
