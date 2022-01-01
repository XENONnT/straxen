
import re
import pymongo
import strax
import pandas as pd

from typing import Any, Type, Union
from .schema import BaseSchema, InsertionError
from .utils import singledispatchmethod

export, __all__ = strax.exporter()

@export
class RemoteDataframe:
    schema: Type[BaseSchema]
    db: Any
    
    def __init__(self, schema, db):
        if isinstance(db, str):
            if db.startswith('mongodb'):
                db = pymongo.MongoClient(db)
            elif db.endswith('.csv'):
                db = pd.read_csv(db)
            elif db.endswith('.pkl'):
                db = pd.read_pickle(db)
            elif db.endswith('.pq'):
                db = pd.read_parquet(db)
            else:
                raise TypeError("Unsupported database type")
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

    def sel_records(self,  *args, **kwargs):
        return self.schema.index.query_db(self.db, *args, **kwargs)

    def sel_record(self, *args, **kwargs):
        records = self.sel_records(*args, **kwargs)
        if records:
            return records[0]       
        raise KeyError('Selection returned no records.')

    def head(self, n=10):
        docs = self.schema.index.head(self.db, n)
        index_fields = self.schema.index_names()
        df = pd.DataFrame(docs, columns=self.schema.all_fields())
        idx = [c for c in index_fields if c in df.columns]
        return df.sort_values(idx).set_index(idx)

    def sel(self, *args, **kwargs):
        docs = self.sel_records(*args, **kwargs)
        index_fields = self.schema.index_names()
        df = pd.DataFrame(docs, columns=self.schema.all_fields())
        idx = [c for c in index_fields if c in df.columns]
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

    def insert(self, records: Union[pd.DataFrame,list]):
        if isinstance(records, pd.DataFrame):
            records = records.reset_index().to_dict(orient='records')
        succeeded = []
        failed = []
        errors = []
        for record in records:
            doc = self.schema(**record)
            try:
                doc.save(self.db, **record)
                succeeded.append(doc.dict())
            except InsertionError as e:
                failed.append(doc.dict())
                errors.append(str(e))

        return succeeded, failed, errors

    def __dir__(self):
        return self.columns + super().__dir__()

    def __getattr__(self, name):
        if name in self.columns:
            return self[name]
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"RemoteDataFrame(index={self.schema.index_names()}, columns={self.schema.columns()})"

class RemoteSeries:
    obj: RemoteDataframe
    column: str

    def __init__(self, obj, column):
        self.obj = obj
        self.column = column

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        return self.obj.sel(*index)[self.column]

    def sel(self, *args, **kwargs):
        df = self.obj.sel(*args, **kwargs)
        return df[self.column]

    def sel_values(self,  *args, **kwargs):
        docs = self.obj.sel_records(*args, **kwargs)
        return [doc[self.column] for doc in docs]

    def sel_value(self, *args, **kwargs):
        values = self.sel_values(*args, **kwargs)
        if values:
            return values[0]
        raise KeyError('Selection returned no values.')

    def set(self, *args, **kwargs):
        raise InsertionError('Cannot set values on a RemoteSeries object,'
                             'use the RemoteDataFrame.')

    def __repr__(self) -> str:
        return f"RemoteSeries(index={self.obj.schema.index_names()}, column={self.column})"


class Indexer:
    def __init__(self, obj: RemoteDataframe):
        self.obj = obj


class LocIndexer(Indexer):

    def __getitem__(self, index):
        columns = None
        
        if isinstance(index, tuple) and len(index) == 2:
            index, columns = index
            if not isinstance(columns, list):
                columns = [columns]
            if not all([c in self.obj.columns for c in columns]):
                if not isinstance(index, tuple):
                    index = (index,)
                index = index + tuple(columns)
                columns = None

        elif isinstance(index, tuple) and len(index) == len(self.obj.columns)+1:
            index, columns = index[:-1], index[-1]

        if not isinstance(index, tuple):
            index = (index,)
        
        df = self.obj.sel(*index)
    
        if columns is not None:
            df = df[columns]

        return df

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        if not isinstance(value, dict):
            value = {'value': value}
        self.obj.set(*key, **value)


class AtIndexer(Indexer):

    def __getitem__(self, key):
        
        if not (isinstance(key, tuple) and len(key)==2):
            raise KeyError('ill-defined location. Specify '
                           '.at[index,column] where index can be a tuple.')
        
        index, column = key

        if column not in self.obj.columns:
            raise KeyError(f'{column} not found. Valid columns are: {self.obj.columns}')
        
        if not isinstance(index, tuple):
            index = (index,)

        if any([isinstance(idx, (slice, list, type(None))) for idx in index]):
            raise KeyError(f'{index} is not unique index.')

        if len(index)<len(self.obj.schema.index_names()):
            KeyError(f'{index} is an under defined index.')

        return self.obj[column].sel_value(*index)
