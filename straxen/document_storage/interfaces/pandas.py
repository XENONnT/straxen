import strax

import pandas as pd

from .. import Index
from .. import IntervalIndex
from .. import InterpolatedIndex

export, __all__ = strax.exporter()


@Index.build_query.register(pd.core.generic.NDFrame)
def build_pandas_query(self, db, value):
    return f'{self.name}==@{self.name}', {self.name: value}

@Index.apply_query.register(pd.DataFrame)
def apply_dataframe(self, db, query):
    if not isinstance(query, list):
        query = [query]
    for q in query:
        if isinstance(q, tuple) and len(q)==2:
            kwargs = q[1]
            q = q[0]
        else:
            kwargs = {}
        if not q:
            continue
        db = db.query(q, local_dict=kwargs)
    if any(db.index.names):
        db = db.reset_index()
    docs = db.to_dict(orient='records')
    return docs

@Index.apply_query.register(pd.Series)
def apply_series(self, db, query):
    return self.apply_query(db, query.to_frame())


@InterpolatedIndex.build_query.register(pd.core.generic.NDFrame)
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


@IntervalIndex.build_query.register(pd.core.generic.NDFrame)
def build_pandas_query(self, db, intervals):
    if isinstance(intervals, list):
        return [pandas_overlap_query(self, iv) for iv in intervals]
    else:
        return pandas_overlap_query(self, intervals)

def pandas_overlap_query(index, interval):
    gt_op = '>=' if index.closed in ['right', 'both'] else '>'
    lt_op = '<=' if index.closed in ['left', 'both'] else '<'
    if isinstance(interval, tuple):
        left, right = interval
    elif isinstance(interval, slice):
        left, right = interval.start, interval.stop
    else:
        left = right = interval
    conditions = []
    if left is not None:
        condition = f'{index.right_name}{gt_op}@{index.left_name}'
        conditions.append(condition)
    if right is not None:
        condition = f'{index.left_name}{lt_op}@{index.right_name}'
        conditions.append(condition)
    query = " and ".join(conditions)
    return query, {index.left_name: left, index.right_name: right}

