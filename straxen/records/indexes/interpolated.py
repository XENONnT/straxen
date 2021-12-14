
from typing import Callable, Union

import datetime
import pymongo
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .index import Index
from ..utils import singledispatchmethod, singledispatch


def nn_interpolate(x, xs, ys):
    idx = np.argmin(np.abs(x-np.array(xs)))
    return ys[idx]

@singledispatch
def interpolater(x, xs, ys, kind='linear'):
    raise f"Interpolation on type {type(x)} is not supported."

@interpolater.register(float)
@interpolater.register(int)
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

class InterpolatedIndex(Index):
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

    def validate(self, value):
        if self.coerce is not None:
            value = self.coerce(value)
        if not isinstance(value, self.type):
            raise TypeError(f'{self.name} must be of type {self.type}')

    def can_extrapolate(self, index):
        if callable(self.extrapolate):
            return self.extrapolate(index)
        return self.extrapolate

    def reduce(self, docs, value):
        if value is None:
            return docs
        new_record = {self.name: value}
        if len(docs)==1:
            new_record.update(docs[0])
            if self.can_extrapolate(new_record):
                new_record[self.name] = value
            else:
                return []
        else:
            xs = [d[self.name] for d in docs]
            for yname in docs[0]:
                ys = [d[yname] for d in docs]
                new_record[yname] = interpolater(value, 
                                                xs, ys, kind=self.kind)
        return [new_record]

    @singledispatchmethod
    def build_query(self, db, value):
        raise TypeError(f"{type(db)} backend not supported.")

    @build_query.register(pymongo.common.BaseObject)
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

    @build_query.register(pd.core.generic.NDFrame)
    def build_pandas_query(self, db, value):
        queries = []
        kwargs = {}
        idx_column = db.reset_index()[self.name]
        before = idx_column[idx_column<=value]
        if len(before):
            before_idx = before.iloc[(before-value).abs().argmin()]
            query = f"({self.name}==@{self.name}__before)"
            kwargs[f"{self.name}__before"] = before_idx
            queries.append(query)
        after = idx_column[idx_column>value]
        if len(after):
            after_idx = after.iloc[(after-value).abs().argmin()]
            query = f"({self.name}==@{self.name}__after)"
            kwargs[f"{self.name}__after"] = after_idx
            queries.append(query)
        return " or ".join(queries), kwargs
