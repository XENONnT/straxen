
import strax
import pandas as pd

from typing import Any, Type
from .document import BaseDocument

export, __all__ = strax.exporter()

@export
class DocumentClient:
    document: Type[BaseDocument]
    db: Any
    
    def __init__(self, document, db):
        self.document = document
        self.db = db
        
    def get(self, *args, **kwargs):
        docs = self.document.index.query_db(self.db, *args, **kwargs)
        return docs
            
    def get_df(self, *args, **kwargs):
        docs = self.get(*args, **kwargs)
        df = pd.DataFrame(docs, columns=list(self.document.all_fields()))
        idx = [c for c in self.document.index.query_fields if c in df.columns]
        return df.sort_values(idx).set_index(idx)

    def set(self, *args, **kwargs):
        doc = self.document(**kwargs)
        return doc.save(self.db, *args, **kwargs)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        docs = self.get(*index)
        nfields = len(self.document.index.query_fields)
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

