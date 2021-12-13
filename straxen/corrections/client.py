
from typing import Any
import pytz
import time
import strax
import straxen
import utilix

import pandas as pd


export, __all__ = strax.exporter()

@export
class CorrectionsClient:
    db: Any

    def __init__(self, db):
        self.db = db
    
    @classmethod
    def default(cls):
        import pymongo
        db = pymongo.MongoClient()['cmt2']
        return cls(db)

    @property
    def corrections(self):
        return straxen.BaseCorrection.subclasses()
    
    @property
    def correction_names(self):
        return list(self.corrections)

    def client(self, name):
        correction = self.corrections[name]
        return straxen.RecordClient(correction, self.db)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key[0] in self.corrections:
            return self.client(key[0])[key[1:]]
        
        if key in self.corrections:
            return self.client(key)
        raise KeyError(key)
        
    def __dir__(self):
        return super().__dir__() + list(self.corrections)
    
    def __getattr__(self, name):
        if name in self.corrections:
            return self[name]
        raise AttributeError(name)

    def get(self, correction_name, *args, **kwargs):
        return self.client(correction_name).get(*args, **kwargs)

    def set(self, correction_name, **kwargs):
        return self.client(correction_name).set(**kwargs)

        
def run_id_to_time(run_id):
    run_id = int(run_id)
    runsdb = utilix.rundb.xent_collection()
    rundoc = runsdb.find_one(
            {'number': run_id},
            {'start': 1})
    if rundoc is None:
        raise ValueError(f'run_id = {run_id} not found')
    time = rundoc['start']
    return time.replace(tzinfo=pytz.utc)

def infer_time(kwargs):
    if 'time' in kwargs:
        return pd.to_datetime(kwargs['time'], utc=True)
    if 'run_id' in kwargs:
        return run_id_to_time(kwargs['run_id'])
    else:
        return None
