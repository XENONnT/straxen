import pandas as pd
import datetime
import numpy as np
from scipy.interpolate import interp1d
from typing import Callable, Union, Type

class Indexer:
    name: str = 'indexer'
    type: Type = (str,int,tuple,float,datetime.datetime)
    coerce: Callable = None

    def __init__(self, type=None, coerce=None):
        if type is not None:
            self.type = type
        self.coerce = coerce
            
    def __set_name__(self, owner, name):
        if '_indices' not in vars(owner):
            owner._indices = {}
        owner._indices[name] = self
        self.name = name

    @property
    def query_fields(self):
        return (self.name, )
        
    @property
    def store_fields(self):
        return (self.name,)

    @property
    def fields(self):
        return (self.name,)
            
    def construct_index(self, record):
        return record.get(self.name, None)
    
    def process_records(self, records, index_value):
        return records

    def validate(self, value):
        if self.coerce is not None:
            value = self.coerce(value)
        if not isinstance(value, self.type):
            raise TypeError(f'{self.name} must be of type {self.type}')