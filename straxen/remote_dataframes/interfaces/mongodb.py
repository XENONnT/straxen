"""
Interface to mongodb databases.
These functions are dispatched when an operation
is being applied to a pymongo object.
"""

import pymongo
import strax
from ..indexes import BaseIndex, IntervalIndexMixin, InterpolatedIndexMixin
from ..schema import BaseSchema, InsertionError

export, __all__ = strax.exporter()


@BaseSchema._insert.register(pymongo.collection.Collection)
def _save_mongo_collection(self, db, index, doc):
    '''We want the client logic to be agnostic to
    whether the value being replaced is actually stored in the DB or
    was inferred from e.g interpolation. 
    The find_one_and_replace(upsert=True) logic is the best match
    for the behavior we want even though it wasts an insert operation
    when a document already exists.
    FIXME: Maybe we can optimize this with an pipeline to
     avoid replacing existing documents with a copy.
    '''
    try:
        doc = db.find_one_and_update(index, {'$set': doc}, projection={'_id': False},
                     upsert=True, return_document=pymongo.ReturnDocument.AFTER)
        return doc
    except Exception as e:
        raise InsertionError(f"Mongodb has rejected this insertion:\n {e} ")

@BaseSchema._insert.register(pymongo.database.Database)
def _save_mongo_database(self, db, index, doc):
    '''If a mongo database was passed to the insert operation
    the correct collection needs to be passed instead.
    '''
 
    return self._insert(db[self.name], index, doc)

@BaseIndex.ensure_index.register(pymongo.collection.Collection)
def ensure_collection_index(self, db):
    db.create_index([(name, pymongo.ASCENDING) for name in self.store_fields],
                    background=True)

@BaseIndex.ensure_index.register(pymongo.database.Database)
def ensure_databse_index(self, db):
    db = db[self.schema.name]
    self.ensure_index(db)

@BaseIndex.head.register(pymongo.collection.Collection)
def collection_head(self, db, n):
    '''replicate pandas .head method, returns the first n documents
    '''
    return list(db.find(projection={'_id': 0}).limit(n))


@BaseIndex.head.register(pymongo.database.Database)
def database_head(self, db, n):
    '''If a mongo database was passed to the head operation
    the correct collection needs to be passed instead.
    '''
    return self.head(db[self.schema.name], n)


@BaseIndex.apply_query.register(pymongo.collection.Collection)
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


@BaseIndex.apply_query.register(pymongo.database.Database)
def apply_mongo_query(self, db, query):
    '''If a database was passed to this operation
    its applied to a collection instead
    '''
    return self.apply_query(db[self.schema.name], query)


@BaseIndex.build_query.register(pymongo.common.BaseObject)
def build_mongo_query(self, db, value):
    '''Simple index matches on equality
    if this index was omited, match all.
    '''
    if isinstance(value, slice):
        # support basic slicing, this will only work
        # for values that are comparable with the
        #  $gt/$lt operators
        start = value.start
        stop = value.stop
        value = {}
        if start is not None:
            value['$gte'] = start
        if stop is not None:
            value['$lt'] = stop
        if not value:
            value = None

    elif isinstance(value, list):
        # support querying multiple values
        # in the same request
        value = {'$in': value}

    return [{
            '$match': {self.name: value} if value is not None else {},
            },
            {
            '$project': {"_id": 0,},
            },
            ]


@InterpolatedIndexMixin.build_query.register(pymongo.common.BaseObject)
def build_interpolation_query(self, db, values):
    '''For interpolation we match the values directly before and after
    the value of interest. For each value we take the closest document on either side.
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
                # support multiple independent aggregations
                # using the facet feature 
                '$facet': queries,
            },

            # Combine results of all aggregations 
            {
                '$project': {
                    'union': {
                        '$setUnion': [f'${name}' for name in queries],
                        }
                        }
            },
            # we just want a single list of documents
            {
                '$unwind': '$union'
            },
            # move list of documents to the root of the result
            # so we just get a nice list of documents
            {
                '$replaceRoot': { 'newRoot': "$union" }
            },
        ]        


@IntervalIndexMixin.build_query.register(pymongo.common.BaseObject)
def build_interval_query(self, db, intervals):
    '''Query overlaping documents with given interval, supports multiple 
    intervals as well as zero length intervals (left==right)
    multiple overlap queries are joined with the $or operator 
    '''
    if not isinstance(intervals, list):
        intervals = [intervals]
    
    queries = []
    for interval in intervals:
        if interval is None:
            continue
        if isinstance(interval, tuple) and all([i is None for i in interval]):
            continue
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
                # support querying for multiple values
                # in a single pipeline
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

    Overlap definition:
    The two intervals (L,R) and (l,r) overlap iff L<r and l<R
    The operator < is replaced with <= when the interval is closed on that side.
    Where if L/l are None, they are treated as -inf
    and if R/r are None, the are treated as inf 
    '''

    # Set the appropriate operators depending on if the interval
    # is closed on one side or both
    gt_op = '$gte' if index.closed in ['right', 'both'] else '$gt'
    lt_op = '$lte' if index.closed in ['left', 'both'] else '$lt'

    # handle different kinds of interval definitions
    if isinstance(interval, tuple):
        left, right = interval
    elif isinstance(interval, slice):
        left, right = interval.start, interval.stop
    else:
        left = right = interval
    
    # Some conditions may not apply if the query interval is None
    # on one or both sides
    conditions = []
    if left is not None:
        conditions.append(
                {   
                    '$or': [
                        # if the right side of the queried interval is
                        # None, treat it as inf
                        {index.right_name: None},
                        {index.right_name: {gt_op: left}},
                        ]
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
                # Add a field splitting the documents into
                # before and after the value of interest
                '_after': {'$gte': [f'${name}', value]},

                # Add a field with the distance to the value of interest
                '_diff': {'$abs': {'$subtract': [value, f'${name}']}},        
                }
        },
        {
            # sort in ascending order by distance
            '$sort': {'_diff': 1},
        },
        {
            # first group by whether document is before or after the value
            # the take the first document in each group
            '$group' : { '_id' : '$_after', 'doc': {'$first': '$$ROOT'},  }
        },
        {
            # make the documents the new root, discarding the groupby value
            "$replaceRoot": { "newRoot": "$doc" },
        },
        {
            # drop the extra fields, they are no longer needed
            '$project': {"_id": 0, '_diff':0, '_after':0 },
        },
    ]
