
from typing import Callable, Union

from pymongo import cursor
import toolz
import datetime
import pymongo
import numpy as np

from dask.utils import Dispatch
from scipy.interpolate import interp1d

from .indexer import Indexer

def nn_interpolate(x, xs, ys):
    idx = np.argmin(np.abs(x-np.array(xs)))
    return ys[idx]

interpolater = Dispatch('interpolate')

@interpolater.register((float, int))
def interpolate_number(x, xs, ys, kind='linear'):
    if isinstance(ys[0], (float, int)):
        func = interp1d(xs, ys, fill_value=(ys[0], ys[-1]),
                        bounds_error=False, kind=kind)
        return np.asscalar(func(x))
    return nn_interpolate(x,xs,ys)

@interpolater.register(datetime.datetime)
def interpolate_datetime(x, xs, ys, kind='linear'):
    xs = [x.timestamp() for x in xs]
    x = x.timestamp()
    if isinstance(ys[0], (float, int)):
        return interpolate_number(x, xs, ys, kind=kind)
    return nn_interpolate(x,xs,ys)

class InterpolatedIndexer(Indexer):
    kind: str
    neighbours: int
    inclusive: bool
    extrapolate: Union[bool,Callable]

    def __init__(self, kind='linear', neighbours=1, 
                inclusive=False, extrapolate=False, **kwargs):
        super().__init__(**kwargs)
        self.kind = kind
        self.neighbours = neighbours
        self.inclusive = inclusive
        self.extrapolate = extrapolate
    
    def can_extrapolate(self, index):
        if callable(self.extrapolate):
            return self.extrapolate(index)
        return self.extrapolate

    def process_records(self, records, index):
        if self.name not in index:
            return records
        keys = list(index)
        keys.remove(self.name)
        new_records = []
        for values,grp in toolz.groupby(keys, records).items():
            new_record = dict(zip(keys, values))
            if len(grp)==1:
                if self.can_extrapolate(index):
                    new_record.update(grp[0])
                    new_record[self.name] = index[self.name]
                    new_records.append(new_record)
                continue

            xs = [r[self.name] for r in grp]
            for yname in grp[0]:
                if yname in index:
                    continue
                ys = [r[yname] for r in grp]
                new_record[yname] = interpolater(index[self.name],
                                                xs, ys, kind=self.kind)
            new_record[self.name] = index[self.name]
            
            new_records.append(new_record)
        return new_records

    def query_db(self, db, key, value):
        return self.apply_selection(db, key, value, self.neighbours)

    def reduce(self, name, docs, **index):
        new_record = dict(index)
        if len(docs)==1:
            if self.can_extrapolate(index):
                new_record.update(docs[0])
                new_record[name] = index[name]
            else:
                return []
        else:
            xs = [d[name] for d in docs]
            for yname in docs[0]:
                if yname in index:
                    continue
                ys = [d[yname] for d in docs]
                new_record[yname] = interpolater(index[name], 
                                                xs, ys, kind=self.kind)
            new_record[name] = index[name]
        return [new_record]

    def process(self, key, value, docs):
        for doc in docs:
            doc[key] = value
        return docs

@InterpolatedIndexer.apply_selection.register(pymongo.collection.Collection)
def mongo_collection(db, key, value, limit=1):
    return InterpolatedIndexer.apply_selection(db.find(), key, value, limit=limit)

@InterpolatedIndexer.apply_selection.register(pymongo.cursor.Cursor)
def mongo_cursor(db, key, value, limit=1):
    after_cursor = db.clone()
    cursors = []

    after_cursor._Cursor__spec[key] = {"$gt": value}
    before_cursor = after_cursor.sort(key, pymongo.ASCENDING).limit(limit)

    cursors.append(after_cursor)
    before_cursor = db.clone()
    before_cursor._Cursor__spec[key] = {"$lte": value}
    before_cursor = before_cursor.sort(key, pymongo.DESCENDING).limit(limit)
    cursors.append(before_cursor)
    return cursors

@InterpolatedIndexer.apply_selection.register(list)
def apply_list(db, key, value, limit=1):
    return [c for d in db for c in InterpolatedIndexer.apply_selection(d, key, value, limit=limit)]

@InterpolatedIndexer.apply_selection.register(dict)
def apply_dict( db,key, value, limit=1):
    return {k: InterpolatedIndexer.apply_selection(d,key, value, limit=limit) for k,d in db.items()}
