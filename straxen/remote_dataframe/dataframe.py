
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
            return ColumnsClient(self, index)
        if isinstance(index, list) and all([idx in self.columns for idx in index]):
            return ColumnsClient(self, index)
        if not isinstance(index, tuple):
            index = (index,)
        docs = self.get(*index)
        nfields = len(self.schema.index.query_fields)
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

class ColumnsClient:
    client: RemoteDataframe
    column: Union[list,str]

    def __init__(self, client, column):
        self.client = client
        self.column = column

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        docs = self.client.get(*index)
        if isinstance(self.column, str):
            docs = [doc[self.column] for doc in docs]
        else:
            docs = [{k: doc[k] for k in self.column} for doc in docs]
        if len(docs)==1:
            return docs[0]
        return docs
