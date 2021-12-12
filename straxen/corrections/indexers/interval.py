import pandas as pd
import pymongo

from .indexer import Indexer

class IntervalIndexer(Indexer):
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

    def query_db(self, db, key, value):
        return self.apply_selection(db, value, 
                                    left_name=self.left_name,
                                    right_name=self.right_name,
                                    closed=self.closed)

    def process(self, key, value, docs):
        for doc in docs:
            doc[key] = value
        return docs


@IntervalIndexer.apply_selection.register(pymongo.collection.Collection)
def mongo_collection(db, value, left_name, right_name, closed):
    return IntervalIndexer.apply_selection(db.find(), value, left_name, right_name, closed)

@IntervalIndexer.apply_selection.register(pymongo.cursor.Cursor)
def mongo_cursor(db, value, left_name, right_name, closed):
    if isinstance(value, tuple) and len(value)==2:
        left, right = value
    elif isinstance(value, slice):
        left, right = value.start, value.stop
    else:
        left = right = value
    if left>right:
        left, right = right, left
    rquery = {}
    right_op = '$gte' if closed in ['right', 'both'] else '$gt'
    rquery = {'$or': [
        {right_name: None},
        {right_name: {right_op: left}}
                     ]}
    left_op = '$lte' if closed in ['left', 'both'] else '$lt'
    lquery = {'$or': [
        {left_name: None},
        {left_name: {left_op: right}}
    ]}
    if '$and' in db._Cursor__spec:
        db._Cursor__spec['$and'].extend([lquery, rquery])
    else:
        db._Cursor__spec['$and'] =  [lquery, rquery]
    return db.clone()

@IntervalIndexer.apply_selection.register(list)
def apply_list(db, value, left_name, right_name, closed):
    return [IntervalIndexer.apply_selection(d, value, left_name, right_name, closed) for d in db]

@IntervalIndexer.apply_selection.register(dict)
def apply_dict( db, value, left_name, right_name, closed):
    return {k: IntervalIndexer.apply_selection(d, value, left_name, right_name, closed) for k,d in db.items()}
