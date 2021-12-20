import strax

import pandas as pd

from .. import Index
from .. import IntervalIndex
from .. import InterpolatedIndex

export, __all__ = strax.exporter()


@Index.build_query.register(pd.core.generic.NDFrame)
def build_pandas_query(self, db, value):
    '''Simple index matches on equality
    if this index was omited, match all.
    these arguments are meant to be used for the df.query()
    the namspace set by the dictionary is available via the @ operator
    when the query is applied, the column names and index names
    are also available by default in the namespace

    example:
      A dataframe with an index named `version` and a user query of
      version=1 will result in the following querybeing applied to
      the dataframe: 

            df.query("version==@version", locals_dict={"version": 1})

      which will select all values with version=1

    '''
    if value is None:
        return ''
    
    return f'{self.name}==@{self.name}', {self.name: value}

@Index.apply_query.register(pd.DataFrame)
def apply_dataframe(self, db, queries):
    ''' Apply one or more queries to a dataframe
    if a query is a tuple of size 2 its assumed that the first
    item is the query and the second is a dict with the namespace 
    to make available to the query eval operation.
    '''
    if not isinstance(queries, list):
        queries = [queries]
    for q in queries:
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
    '''Series dont support the .query() api
    so convert to a databrame and apply query.
    '''
    return self.apply_query(db, query.to_frame())


@InterpolatedIndex.build_query.register(pd.core.generic.NDFrame)
def build_pandas_query(self, db, values):
    '''Interpolated index selects the rows before and after
    the requested value if they exist.
    '''
    
    if not isinstance(values, list):
        values = [values]
    values  = [v for v in values if v is not None]
    queries = []
    kwargs = {}

    # if the column we are matching on is in the index, reset index
    if self.name in db.index.names:
        db = db.reset_index()

    # we only need the column we are matching on
    idx_column = db[self.name]
    for i, value in enumerate(values):

        # select all values before requested values
        before = idx_column[idx_column<=value]

        if len(before):
            # if ther are values after `value`, we find the closest
            # one and add a query for that value.
            before_idx = before.iloc[(before-value).abs().argmin()]
            query = f"({self.name}==@{self.name}__before_{i})"
            kwargs[f"{self.name}__before_{i}"] = before_idx
            queries.append(query)

        # select all values after requested values
        after = idx_column[idx_column>value]
        if len(after):
            # same as before
            after_idx = after.iloc[(after-value).abs().argmin()]
            query = f"({self.name}==@{self.name}__after_{i})"
            kwargs[f"{self.name}__after_{i}"] = after_idx
            queries.append(query)
    
    # match on both before and after queries
    return " or ".join(queries), kwargs


@IntervalIndex.build_query.register(pd.core.generic.NDFrame)
def build_pandas_query(self, db, intervals):
    '''Query by overlap, if multiple overlaps are 
    given, joing them with an or logic
    '''
    if not isinstance(intervals, list):
        intervals = [intervals]

    gt_op = '>=' if self.closed in ['right', 'both'] else '>'
    lt_op = '<=' if self.closed in ['left', 'both'] else '<'

    queries = []
    kwargs = {}

    for idx,interval in enumerate(intervals):
        left, right = self.to_interval(interval)
    
        conditions = []
        if left is not None:
            condition = f'{self.right_name}{gt_op}@{self.left_name}_{idx}'
            conditions.append(condition)
        if right is not None:
            condition = f'{self.left_name}{lt_op}@{self.right_name}_{idx}'
            conditions.append(condition)

        query = " and ".join(conditions)
        query = f"({query})"
        queries.append(query)

        kwargs[f'{self.left_name}_{idx}'] = left
        kwargs[f'{self.right_name}_{idx}'] = right

    query = " or ".join(queries)
    return query, kwargs

