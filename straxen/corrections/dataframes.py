
from typing import Any
import pytz
import strax
import straxen
import utilix

import pandas as pd


export, __all__ = strax.exporter()
__all__ += ['corrections_db']

@export
class CorrectionDataframes:
    db: Any

    def __init__(self, db):
        self.db = db
    
    @classmethod
    def default(cls, *args, **kwargs):
        return cls.local_mongo(*args, **kwargs)
    
    @classmethod
    def local_mongo(cls, dbname='cmt2'):
        import pymongo
        db = pymongo.MongoClient()[dbname]
        return cls(db)

    @property
    def corrections(self):
        return dict(straxen.BaseCorrectionSchema._CORRECTIONS)
    
    @property
    def correction_names(self):
        return list(self.corrections)

    def get_df(self, name):
        correction = self.corrections[name]
        return straxen.RemoteDataframe(correction, self.db)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key[0] in self.corrections:
            return self.get_df(key[0])[key[1:]]
        
        if key in self.corrections:
            return self.get_df(key)
        raise KeyError(key)
        
    def __dir__(self):
        return super().__dir__() + list(self.corrections)
    
    def __getattr__(self, name):            
        if name in self.corrections:
            return self.get_df(name)
        raise AttributeError(name)

    def sel(self, correction_name, *args, **kwargs):
        return self.get_df(correction_name).sel(*args, **kwargs)

    def set(self, correction_name, *args, **kwargs):
        return self.get_df(correction_name).set(*args, **kwargs)

    def insert(self, correction_name, records):
        return self.get_df(correction_name).insert(records=records)
        
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

def extract_time(kwargs):
    if 'time' in kwargs:
        return pd.to_datetime(kwargs.pop('time'), utc=True)
    if 'run_id' in kwargs:
        return run_id_to_time(kwargs.pop('run_id'))
    else:
        return None

corrections_db = CorrectionDataframes.local_mongo()

@export
def cmt2(name, version=0, **kwargs):
    dtime = extract_time(kwargs)
    docs = corrections_db[name].sel(time=dtime, version=version, **kwargs)
    if len(docs)==1:
        return docs[0]
    return docs
