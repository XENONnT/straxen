
import strax
import pandas as pd

from typing import Any, Type
from .record import BaseRecord

export, __all__ = strax.exporter()

@export
class RecordClient:
    record: Type[BaseRecord]
    db: Any
    
    def __init__(self, record, db):
        self.record = record
        self.db = db
        
    def get(self, *args, **kwargs):
        docs = self.record.index.query_db(self.db, *args, **kwargs)
        return docs
            
    def get_df(self, *args, **kwargs):
        docs = self.get(*args, **kwargs)
        df = pd.DataFrame(docs, columns=list(self.record.all_fields()))
        idx = [c for c in self.record.index.query_fields if c in df.columns]
        return df.sort_values(idx).set_index(idx)

    def set(self, *args, **kwargs):
        doc = self.record(**kwargs)
        return doc.save(self.db, *args, **kwargs)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        docs = self.get(*index)
        nfields = len(self.record.index.query_fields)
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

