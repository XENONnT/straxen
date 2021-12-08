
import toolz
import datetime
import numpy as np

from scipy.interpolate import interp1d

from .indexer import Indexer

class InterpolatedIndexer(Indexer):
    kind: str
    
    def __init__(self, kind='linear', inclusive=True):
        self.kind = kind
        self.inclusive = inclusive
    
    def interpolate(self, xs, ys, index):
        xs = [x.timestamp() if isinstance(x, datetime.datetime) else x for x in xs]
        if isinstance(index, datetime.datetime):
            index = index.timestamp()
        func = interp1d(xs, ys, fill_value=(ys[0],ys[-1]), bounds_error=False)
        return np.asscalar(func(index))
    
    def process_records(self, records, index):
        if self.name not in index:
            return records
        keys = list(index)
        keys.remove(self.name)
        new_records = []
        for values,grp in toolz.groupby(keys, records).items():
            new_record = dict(zip(keys, values))
            xs = [r[self.name] for r in grp]
            ys = [r['value'] for r in grp]
            new_record[self.name] = index[self.name]
            new_record['value'] = self.interpolate(xs, ys, index[self.name])
            new_records.append(new_record)
        return new_records
