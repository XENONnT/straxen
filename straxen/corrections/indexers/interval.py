import pandas as pd
import pymongo

from .index import Index
from ..utils import singledispatchmethod

class IntervalIndex(Index):
    left_name: str = 'left'
    right_name: str = 'right'
    closed: str = 'left'
        
    def __init__(self, left_name=None,
                 right_name=None, 
                 closed=None, **kwargs):
        super().__init__(**kwargs)
        if left_name is not None:
            self.left_name = left_name
        if right_name is not None:
            self.right_name = right_name
        if closed is not None:
            self.closed = closed

    @property
    def store_fields(self):
        return (self.left_name, self.right_name)
    
    def construct_index(self, record):
        left = record[self.left_name]
        right = record[self.right_name]

        if not isinstance(left, int):
            left = left if left is None else pd.to_datetime(left)
        if not isinstance(right, int):
            right = right if right is None else pd.to_datetime(right)
        if left is None or right is None:
            return (left, right)
        return pd.Interval(left, right,
                          closed=self.closed)

    def reduce(self, docs, value):
        for doc in docs:
            doc[self.name]= value
        return docs

    @singledispatchmethod
    def build_query(self, db, value):
        raise TypeError(f"{type(db)} backend not supported.")

    @build_query.register(pymongo.collection.Collection)
    @build_query.register(pymongo.database.Database)
    def build_mongo_query(self, db, value):
        gt_op = '$gte' if self.closed in ['right', 'both'] else '$gt'
        lt_op = '$lte' if self.closed in ['left', 'both'] else '$lt'
        return [
            {
                '$match':  {
                    '$and': [
                        {
                            '$or': [{self.right_name: None},
                                    {self.right_name: {gt_op: value}}]
                        },
                        {
                            '$or': [{self.left_name: None},
                                    {self.left_name: {lt_op: value}}]
                        },
                        ]
                }
            },
            {
            '$project': {"_id": 0,},
            },
            
        ]
