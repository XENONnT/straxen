import pandas as pd
import datetime
import numpy as np
from scipy.interpolate import interp1d


class Indexer:
    name: str = 'index'
    
    def __set_name__(self, owner, name):
        if '_indices' not in vars(owner):
            owner._indices = {}
        owner._indices[name] = self
        self.name = name
        
    @property
    def fields(self):
        return {self.name}
            
    def construct_index(self, record):
        return record.get(self.name, None)
    
    def process_records(self, records, index_value):
        return records

