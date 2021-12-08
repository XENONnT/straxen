import pandas as pd

from .indexer import Indexer

class IntervalIndexer(Indexer):
    left_name: str = 'left'
    right_name: str = 'right'
    closed: str = 'left'
    
    def __init__(self, left_name=None,
                 right_name=None, 
                 closed=None):
        if left_name is not None:
            self.left_name = left_name
        if right_name is not None:
            self.right_name = right_name
        if closed is not None:
            self.closed = closed

    @property
    def fields(self):
        return {self.left_name, self.right_name}
    
    def construct_index(self, record):
        left = record[self.left_name]
        if not isinstance(left, int):
            left = pd.to_datetime(left)
        right = record[self.right_name]
        if not isinstance(right, int):
            right = pd.to_datetime(right)
        return pd.Interval(left, right,
                          closed=self.closed)
