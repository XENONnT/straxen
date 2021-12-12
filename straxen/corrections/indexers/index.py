
import datetime
import pymongo
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Callable, Union, Type

from ..utils import singledispatchmethod

class Index:
    name: str = 'index'
    type: Type = (str,int,tuple,float,datetime.datetime)
    coerce: Callable = None


    def __init__(self, type=None, name=None, 
                correction=None, coerce=None,
                ):
        if type is not None:
            self.type = type
        self.coerce = coerce
        if name is not None:
            self.name = name
        if correction is not None:
            self.correction = correction
            
    def __set_name__(self, owner, name):
        self.correction = owner
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

    def query_db(self, db, value):
        query = self.build_query(db, value)
        docs = self.apply_query(db, query)
        docs = self.reduce(docs, value)
        return docs

    def reduce(self, docs, value):
        return docs

    @singledispatchmethod
    def build_query(self, db, value):
        raise TypeError(f"{type(db)} backend not supported.")

    @singledispatchmethod
    def apply_query(self, db, query):
        raise TypeError(f"{type(db)} backend not supported.")

    @build_query.register(pymongo.collection.Collection)
    @build_query.register(pymongo.database.Database)
    def build_mongo_query(self, db, value):
        return [{
                '$match': {self.name: value},
                },
                {
                '$project': {"_id": 0,},
                },
                ]
    
    @apply_query.register(pymongo.collection.Collection)
    def apply_mongo_query(self, db, query):
        if isinstance(query, dict):
            query = [query]
        agg = []
        for q in query:
            if isinstance(q, list):
                agg.extend(q)
            elif isinstance(q, dict):
                agg.append(q)
        return list(db.aggregate(agg))

    @apply_query.register(pymongo.database.Database)
    def apply_mongo_query(self, db, query):
        return self.apply_query(db[self.correction.name], query)

    @apply_query.register(list)
    def apply_list(self, db, query):
        return [self.apply_query(d, query) for d in db]

    @apply_query.register(dict)
    def apply_dict(self, db, query):
        return {k: self.apply_query(d, query) for k,d in db.items()}

