
import datetime
from pandas.core.algorithms import isin
import pymongo
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Callable, Union, Type
from ..utils import singledispatchmethod

class Index:
    name: str = ''
    type: Type = (str,int,tuple,float,datetime.datetime)
    coerce: Callable = None

    def __init__(self, type=None, name=None, 
                record=None, coerce=None,
                ):
        if type is not None:
            self.type = type
        self.coerce = coerce
        if name is not None:
            self.name = name
        if record is not None:
            self.record = record
            
    def __set_name__(self, owner, name):
        self.record = owner
        if not self.name:
            self.name = name

    @property
    def query_fields(self):
        return (self.name, )
        
    @property
    def store_fields(self):
        return (self.name,)

    def infer_index_value(self, **kwargs):
        return kwargs.get(self.name, None)

    def infer_index(self, *args, **kwargs):
        index = dict(zip(self.query_fields, args))
        index.update(kwargs)
        return {self.name: self.infer_index_value(**index)}

    def validate(self, value):
        if isinstance(value, list):
            return [self.validate(val) for val in value]
        
        if self.coerce is not None:
            value = self.coerce(value)
        if not isinstance(value, self.type):
            raise TypeError(f'{self.name} must be of type {self.type}')

    def query_db(self, db, *args, **kwargs):
        value = self.infer_index(*args, **kwargs)[self.name]
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

    @build_query.register(pymongo.common.BaseObject)
    def build_mongo_query(self, db, value):
        if value is None:
            return [{
                '$project': {"_id": 0,},
                }]
        if isinstance(value, list):
            value = {'$in': value}
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
        return self.apply_query(db[self.record.name], query)

    @build_query.register(pd.core.generic.NDFrame)
    def build_pandas_query(self, db, value):
        return f'{self.name}==@{self.name}', {self.name: value}

    @apply_query.register(pd.DataFrame)
    def apply_dataframe(self, db, query):
        if not isinstance(query, list):
            query = [query]
        for q in query:
            if isinstance(q, tuple) and len(q)==2:
                kwargs = q[1]
                q = q[0]
            else:
                kwargs = {}
            if not q:
                continue
            db = db.query(q, local_dict=kwargs)
        if db.index.name is not None:
            db = db.reset_index()
        docs = db.to_dict(orient='records')
        return docs

    @apply_query.register(pd.Series)
    def apply_series(self, db, query):
        return self.apply_query(db, query.to_frame())

    @apply_query.register(list)
    def apply_list(self, db, query):
        return [self.apply_query(d, query) for d in db]

    @apply_query.register(dict)
    def apply_dict(self, db, query):
        return {k: self.apply_query(d, query) for k,d in db.items()}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name},type={self.type})"