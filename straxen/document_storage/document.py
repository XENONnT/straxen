import re
import pymongo
import strax
from straxen.document_storage.utils import singledispatchmethod
import utilix

import datetime

import pandas as pd

from copy import copy
from pydantic import BaseModel
from typing import ClassVar, Type, Union

from .indexes import Index, MultiIndex

export, __all__ = strax.exporter()

class InsertionError(Exception):
    pass

def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

def camel_to_snake(name):
  name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

@export
class BaseDocument(BaseModel):
    name: ClassVar = ''
    index: ClassVar
    value: Union[str,int,float]
    
    def __init_subclass__(cls) -> None:
        #FIXME: maybe move this logic into the Index class?
        # Index can be a descriptor that does a dynamic lookup and
        # concatentation of all indexes in parent classes to return
        # the final indexer initialized with the document class.
        indexes = {}
        for base in reversed(cls.mro()):
            if not issubclass(base, BaseDocument):
                continue
            if 'index' not in base.__dict__:
                continue
            if base.index in indexes:
                continue
            if isinstance(base.index, MultiIndex):
                indexes.update({index.name: copy(index) for index in base.index.indexes})
            elif isinstance(base.index, Index):
                indexes[base.index.name] = copy(base.index)
        if not indexes:
            raise AttributeError(f'Document class {cls.__name__} has no index.')
        if len(indexes) == 1:
            cls.index = list(indexes.values())[0]
        else:
            cls.index = MultiIndex(**indexes)
        cls.index.__set_name__(cls, 'index')
        
    @classmethod
    def all_fields(cls):
        return cls.index.store_fields + tuple(cls.schema()['properties'])

    @classmethod
    def subclasses(cls):
        return {c.name: c for c in all_subclasses(cls) if c.name}

    @classmethod
    def db_client(cls, db):
        from .client import DocumentClient
        return DocumentClient(cls, db)

    @classmethod
    def query_db(cls, db, *args, **kwargs):
        return cls.index.query_db(db, *args, **kwargs)

    def pre_insert(self, **index):
        pass

    def pre_update(self, old, **index):
        if old != self:
            message = f'Values already set for {index} are different\
                        than the values you are trying to set.'
            raise IndexError(message)

    def save(self, db, *args, **kwargs):
        kwargs.update(self.dict())
        existing = self.index.query_db(db, *args, **kwargs)
        if len(existing)>1:
            raise InsertionError('Multiple documents exist that match \
                the index you are attempting to save to.')

        index = self.index.infer_index(*args, **kwargs)
        if len(existing) == 1:
            old = self.__class__(**existing[0])
            self.pre_update(old, **index)
        else:
            self.pre_insert(**index)
        doc = self.dict()
        doc.update(index)
        return self._insert(db, doc)

    @singledispatchmethod
    def _insert(self, db, doc):
        raise TypeError(f'Inserts are not supported \
                        for {type(db)} data stores.')
