import pandas as pd

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
    def fields(self):
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
