
from collections import defaultdict
from typing import Type
from immutabledict import immutabledict
import utilix
import pandas as pd

from .clock import SimpleClock

DEFAULT_DATABASE  ='cmt2'


class CorrectionsSettings:

    cmt_database: str = DEFAULT_DATABASE

    clock = SimpleClock()

    datasources = {}
    
    def default_datasource(self, name):
        return utilix.xent_collection(collection=name,
                                    database=self.cmt_database)

    def get_datasource_for(self, name):
        if name in self.datasources:
            return self.datasources[name]
        return self.default_datasource(name)
        
    def run_id_to_time(self, run_id):
        rundb = utilix.xent_collection()
        if isinstance(run_id, str):
            query = {'number': int(run_id)}
        elif isinstance(run_id, int):
            query = {'number': run_id}
        else:
            raise TypeError(f'Invalid type for run id: {type(run_id)}')

        doc = rundb.find_one(query, projection={'start': 1, 'end': 1})
        if not doc:
            raise KeyError(f'Run {run_id} not found.')

        return doc['start'] + (doc['end']  - doc['start'] ) / 2

    def extract_time(self, kwargs):
        if 'time' in kwargs:
            time = kwargs.pop('time')
        if 'run_id' in kwargs:
            time = self.run_id_to_time(kwargs.pop('run_id'))
        else:
            return None
        return pd.to_datetime(time)

corrections_settings = CorrectionsSettings()
