
from typing import Callable, Union
import toolz
import datetime
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

    def __init__(self, kind='linear', neighbours=1, inclusive=True, extrapolate=False, **kwargs):
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
                new_record[yname] = interpolater(index[self.name], xs, ys, kind=self.kind)
            new_record[self.name] = index[self.name]
            
            new_records.append(new_record)
        return new_records

