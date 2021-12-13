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
    
    def infer_index_value(self, **kwargs):
        if self.name in kwargs:
            value = kwargs[self.name]
            if isinstance(value, tuple):
                left, right = value
            elif isinstance(value, slice):
                left, right = value.start, value.stop 
            else:
                left = right = value

        if self.left_name in kwargs:
            left = kwargs[self.left_name]
        else:
            left = None

        if self.right_name in kwargs:
            right = kwargs[self.right_name]
        else:
            right = None
        return (left, right)

    @singledispatchmethod
    def build_query(self, db, value):
        raise TypeError(f"{type(db)} backend not supported.")

    @build_query.register(pymongo.common.BaseObject)
    def build_mongo_query(self, db, value):
        gt_op = '$gte' if self.closed in ['right', 'both'] else '$gt'
        lt_op = '$lte' if self.closed in ['left', 'both'] else '$lt'
        if isinstance(value, tuple):
            left, right = value
        elif isinstance(value, slice):
            left, right = value.start, value.stop
        else:
            left = right = value
        matchers = []
        if left is not None:
            matchers.append(
                {
                    '$or': [{self.right_name: None},
                            {self.right_name: {gt_op: left}}]
                }
            )
        if right is not None:
            matchers.append(
                {
                    '$or': [{self.left_name: None},
                            {self.left_name: {lt_op: right}}]
                }

            )
        if not matchers:
            return [{
            '$project': {"_id": 0,},
            }]
        return [
            {
                '$match':  {
                    '$and': matchers,
                }
            },
            {
            '$project': {"_id": 0,},
            },
        ]
