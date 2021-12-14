import pymongo
import strax
from .. import Index
from .. import IntervalIndex
from .. import InterpolatedIndex
from .. import BaseDocument


export, __all__ = strax.exporter()


@BaseDocument._insert.register(pymongo.collection.Collection)
def _save_mongo_collection(self, db, doc):
    return db.find_one_and_replace(doc, doc, upsert=True)

@BaseDocument._insert.register(pymongo.database.Database)
def _save_mongo_database(self, db, doc):
    return self._insert(db[self.name], doc)

    
@Index.apply_query.register(pymongo.collection.Collection)
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

@Index.apply_query.register(pymongo.database.Database)
def apply_mongo_query(self, db, query):
    return self.apply_query(db[self.document.name], query)

@Index.build_query.register(pymongo.common.BaseObject)
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

@InterpolatedIndex.build_query.register(pymongo.common.BaseObject)
def build_mongo_query(self, db, value):
    if value is None:
        return [{
            '$project': {"_id": 0},
        }]
    return [
        {
            '$addFields': {
                '_after': {'$gt': [f'${self.name}', value]},
                '_diff': {'$abs': {'$subtract': [value, f'${self.name}']}},        
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


@IntervalIndex.build_query.register(pymongo.common.BaseObject)
def build_mongo_query(self, db, intervals):
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
        