
from datetime import datetime
import re
from pydantic.typing import NoneType
import pymongo
import strax
import pandas as pd

from typing import Any, Dict, List, Type, Union, Tuple
from .schema import BaseSchema, InsertionError
from .utils import singledispatchmethod

export, __all__ = strax.exporter()

Index = Union[int,float,datetime,str,slice,NoneType,List]


@export
class RemoteDataframe:
    '''Implement basic indexing features similar to a pandas dataframe
    but operates on an arbitrary storage backend
    '''
    schema: Type[BaseSchema]
    db: Any

    def __call__(self, column: str, **index: Index) -> pd.DataFrame:
        index = tuple(index.get(k, None) for k in self.index)
        return self.at[index, column]

    def __init__(self, schema: Type[BaseSchema], db: Any, **kwargs) -> None:
        if isinstance(db, str):
            if db.startswith('mongodb'):
                db = pymongo.MongoClient(db, **kwargs)
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

    @classmethod
    def from_mongodb(cls, schema, url, dbname='remote_dataframes', **kwargs):
        import pymongo
        db = pymongo.MongoClient(url, **kwargs)[dbname]
        return cls(schema, db)

    @property
    def name(self):
        return self.schema.name
    
    @property
    def columns(self):
        return self.schema.columns()

    @property
    def index(self):
        return self.schema.index_names()

    @property
    def loc(self):
        return LocIndexer(self)

    @property
    def at(self):
        return AtIndexer(self)

    def sel_records(self, *args: Index, **kwargs: Index) -> List[dict]:
        '''Queries the DB and returns the results as a list of dicts
        '''
        return self.schema.index.query_db(self.db, *args, **kwargs)

    def sel_record(self, *args: Index, **kwargs: Index) -> dict:
        '''Return a single dict
        '''
        records = self.sel_records(*args, **kwargs)
        if records:
            return records[0]       
        raise KeyError('Selection returned no records.')

    def head(self, n=10) -> pd.DataFrame:
        '''Return first n documents as a pandas dataframe
        '''
        docs = self.schema.index.head(self.db, n)
        index_fields = self.schema.index_names()
        df = pd.DataFrame(docs, columns=self.schema.all_fields())
        idx = [c for c in index_fields if c in df.columns]
        return df.sort_values(idx).set_index(idx)

    def sel(self, *args: Index, **kwargs: Index) -> pd.DataFrame:
        '''select a subset of the data
        returns a pandas dataframe
        '''
        docs = self.sel_records(*args, **kwargs)
        index_fields = self.schema.index_names()
        df = pd.DataFrame(docs, columns=self.schema.all_fields())
        idx = [c for c in index_fields if c in df.columns]
        return df.sort_values(idx).set_index(idx)

    def set(self, *args: Index, **kwargs: Index) -> BaseSchema:
        '''Insert data by index
        '''
        doc = self.schema(**kwargs)
        return doc.save(self.db, *args, **kwargs)

    def __getitem__(self, index: Tuple[Index,...]) -> 'RemoteSeries':
        if isinstance(index, str) and index in self.columns:
            return RemoteSeries(self, index)
        if isinstance(index, tuple) and index[0] in self.columns:
            return RemoteSeries(self, index[0])[index[1:]]
        raise KeyError(f'{index} is not a dataframe column.')

    def insert(self, records: Union[pd.DataFrame,List[dict]]) -> Tuple[List[dict],List[dict],List[dict]]:
        ''' Insert multiple records into the DB
        '''
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

    def __dir__(self) -> List[str]:
        return self.columns + super().__dir__()

    def __getattr__(self, name: str) -> 'RemoteSeries':
        if name in self.columns:
            return self[name]
        raise AttributeError(name)

    def __repr__(self) -> str:
        return (f"RemoteDataFrame(name={self.name},"
               f"index={self.schema.index_names()},"
               f"columns={self.schema.columns()})")

class RemoteSeries:
    obj: RemoteDataframe
    column: str

    def __init__(self, obj: RemoteDataframe, column: str) -> None:
        self.obj = obj
        self.column = column

    def __getitem__(self, index: Union[Index,Tuple[Index,...]]) -> pd.DataFrame:
        if not isinstance(index, tuple):
            index = (index,)
        return self.obj.sel(*index)[self.column]

    def sel(self, *args: Index, **kwargs: Index) -> pd.DataFrame:
        df = self.obj.sel(*args, **kwargs)
        return df[self.column]

    def sel_values(self, *args: Index, **kwargs: Index) -> List[Any]:
        docs = self.obj.sel_records(*args, **kwargs)
        return [doc[self.column] for doc in docs]

    def sel_value(self, *args: Index, **kwargs: Index) -> Any:
        values = self.sel_values(*args, **kwargs)
        if values:
            return values[0]
        raise KeyError('Selection returned no values.')

    def set(self,  *args: Index, **kwargs: Index):
        raise InsertionError('Cannot set values on a RemoteSeries object,'
                             'use the RemoteDataFrame.')

    def __repr__(self) -> str:
        return (f"RemoteSeries(index={self.obj.schema.index_names()},"
                f"column={self.column})")


class Indexer:
    def __init__(self, obj: RemoteDataframe):
        self.obj = obj


class LocIndexer(Indexer):

    def __call__(self, *args: Index, **kwargs: Index) -> pd.DataFrame:
        return self.obj.sel(*args, **kwargs)

    def __getitem__(self, index: Tuple[Index]) -> pd.DataFrame:
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

    def __setitem__(self, key: str, value: Union[dict,BaseSchema]) -> BaseSchema:
        if not isinstance(key, tuple):
            key = (key,)
        
        if isinstance(value, self.obj.schema):
            value  = value.dict()

        if not isinstance(value, dict):
            value = {'value': value}

        return self.obj.set(*key, **value)


class AtIndexer(Indexer):

    def __getitem__(self, key: Tuple[Tuple[Index, ...],str]) -> Any:
        
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
