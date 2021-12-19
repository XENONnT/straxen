"""
Interface to mongodb databases.
These functions are dispatched when an operation
is being applied to a pymongo object.
"""

import pymongo
import strax
from .. import Index
from .. import IntervalIndex
from .. import InterpolatedIndex
from .. import BaseDocument


export, __all__ = strax.exporter()


@BaseDocument._insert.register(pymongo.collection.Collection)
def _save_mongo_collection(self, db, doc):
    '''We want the client logic to be agnostic to
    whether the value being replaced is actually stored in the DB or
    was inferred from e.g interpolation. 
    The find_one_and_replace(upsert=True) logic is the best match
    for the behavior we want even though it wasts an insert operation
    when a document already exists.
    FIXME: Maybe we can optimize this with an aggregation to
     avoid replacing existing documents with a copy.
    '''
    return db.find_one_and_replace(doc, doc, upsert=True)

@BaseDocument._insert.register(pymongo.database.Database)
def _save_mongo_database(self, db, doc):
    '''If a mongo database was passed to the insert operation
    the correct collection needs to be passed instead.
    '''
    return self._insert(db[self.name], doc)

    
@Index.apply_query.register(pymongo.collection.Collection)
def apply_mongo_query(self, db, query):
    '''apply one or more mongo queries as
    an aggregation
    '''
    if isinstance(query, dict):
        query = [query]
    agg = []
    for q in query:
        if isinstance(q, list):
            agg.extend(q)
        elif isinstance(q, dict):
            agg.append(q)
    return list(db.aggregate(agg))

@Index.apply_query.register(pymongo.database.Database)
def apply_mongo_query(self, db, query):
    '''If a database was passed to this operation
    its applied to a collection instead
    '''
    return self.apply_query(db[self.document.name], query)

@Index.build_query.register(pymongo.common.BaseObject)
def build_mongo_query(self, db, value):
    '''Simple index matches on equality
    if this index was omited, match all.
    '''
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

@InterpolatedIndex.build_query.register(pymongo.common.BaseObject)
def build_interpolation_query(self, db, values):
    '''For interpolation we match the values directly before and after
    the value of interest
    '''
    if values is None:
        return [{
            '$project': {"_id": 0},
        }]
    if not isinstance(values, list):
        values = [values]
    
    queries = {f'agg{i}': mongo_closest_query(self.name, value) for i,value in enumerate(values)}
    return [
            {
                '$facet': queries,
            },
            {
                '$project': {
                    'union': {
                        '$setUnion': [f'${name}' for name in queries],
                        }
                        }
            },
            {
                '$unwind': '$union'
            },
            {
                '$replaceRoot': { 'newRoot': "$union" }
            },
        ]        


@IntervalIndex.build_query.register(pymongo.common.BaseObject)
def build_interval_query(self, db, intervals):
    '''Query overlaping documents with given interval, supports multiple 
    intervals as well as zero length intervals (left==right)
    multiple overlap queries are joined with the $or operator 
    '''
    if not isinstance(intervals, list):
        intervals = [intervals]
    queries = []
    for interval in intervals:
        query = mongo_overlap_query(self, interval)
        if query:
            queries.append(query)
    
    if not queries:
        return [{
        '$project': {"_id": 0,},
        }]
    return [
        {
            '$match':  {
                '$or': queries,
            }
        },
        {
        '$project': {"_id": 0,},
        },
    ]

def mongo_overlap_query(index, interval):
    '''Builds a single overlap query
    Intervals with one side equal to null are treated as extending to infinity in
    that direction.
    Supports closed or open intervals as well as infinite intervals
    '''
    gt_op = '$gte' if index.closed in ['right', 'both'] else '$gt'
    lt_op = '$lte' if index.closed in ['left', 'both'] else '$lt'
    if isinstance(interval, tuple):
        left, right = interval
    elif isinstance(interval, slice):
        left, right = interval.start, interval.stop
    else:
        left = right = interval
    conditions = []
    if left is not None:
        conditions.append(
                {
                    '$or': [{index.right_name: None},
                            {index.right_name: {gt_op: left}}]
                }
            )
    if right is not None:
        conditions.append(
                {
                    '$or': [{index.left_name: None},
                            {index.left_name: {lt_op: right}}]
                }

            )
    if conditions:
        return {
                '$and': conditions,
            }
    else:
        return {}

def mongo_closest_query(name, value):
    return [
        {
            '$addFields': {
                '_after': {'$gt': [f'${name}', value]},
                '_diff': {'$abs': {'$subtract': [value, f'${name}']}},        
                }
        },
        {
            '$sort': {'_diff': 1},
        },
        {
            '$group' : { '_id' : '$_after', 'doc': {'$first': '$$ROOT'},  }
        },
        {
            "$replaceRoot": { "newRoot": "$doc" },
        },
        {
            '$project': {"_id": 0, '_diff':0, '_after':0 },
        },
    ]
