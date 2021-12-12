
import datetime
import pymongo
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Callable, Union, Type
from dask.utils import Dispatch

class Indexer:
    name: str = 'indexer'
    type: Type = (str,int,tuple,float,datetime.datetime)
    coerce: Callable = None
    apply_selection = Dispatch('apply_selection')

    def __init_subclass__(cls) -> None:
        cls.apply_selection = Dispatch('apply_selection')

    def __init__(self, type=None, coerce=None):
        if type is not None:
            self.type = type
        self.coerce = coerce
            
    def __set_name__(self, owner, name):
        if '_indices' not in vars(owner):
            owner._indices = {}
        owner._indices[name] = self
        self.name = name

    @property
    def query_fields(self):
        return (self.name, )
        
    @property
    def store_fields(self):
        return (self.name,)

    @property
    def fields(self):
        return (self.name,)
            
    def construct_index(self, record):
        return record.get(self.name, None)
    
    def process_records(self, records, index_value):
        return records

    def validate(self, value):
        if self.coerce is not None:
            value = self.coerce(value)
        if not isinstance(value, self.type):
            raise TypeError(f'{self.name} must be of type {self.type}')

    def query_db(self, db, key, value):
        return self.apply_selection(db, key, value)

    def process(self, key, value, docs):
        return docs
        
    def reduce(self, name, docs, **index):
        return docs

@Indexer.apply_selection.register(pymongo.collection.Collection)
def mongo_collection(db, key, value):
    return db.find(filter={key: value})

@Indexer.apply_selection.register(pymongo.cursor.Cursor)
def mongo_cursor(db, key, value):
    db._Cursor__spec[key] = value
    return db

@Indexer.apply_selection.register(list)
def apply_list(db, key, value):
    return [Indexer.apply_selection(d, key, value) for d in db]

@Indexer.apply_selection.register(dict)
def apply_dict(db, key, value):
    return {k: Indexer.apply_selection(d, key, value) for k,d in db.items()}


