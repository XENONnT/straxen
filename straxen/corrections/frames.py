
from typing import Any
import pytz
import strax
import straxen
import utilix

import pandas as pd
import rframe

export, __all__ = strax.exporter()
__all__ += ['cframes']

@export
class CorrectionFrames:
    db: Any

    def __init__(self, db):
        self.db = db
    
    @classmethod
    def default(cls, **kwargs):
        return cls.from_utilix(**kwargs)

    @classmethod
    def from_mongodb(cls, url='localhost', db='cmt2', **kwargs):
        import pymongo
        db = pymongo.MongoClient(url, **kwargs)[db]
        return cls(db)

    @classmethod
    def from_utilix(cls, experiment='xent', db='cmt2'):
        coll = utilix.rundb._collection(collection='dummy',
                                        experiment=experiment,
                                        database=db)
        db = coll.database
        return cls(db)

    @property
    def schemas(self):
        return dict(straxen.BaseCorrectionSchema._SCHEMAS)
    
    @property
    def correction_names(self):
        return list(self.schemas)

    def get_df(self, name):
        correction = self.schemas[name]
        return rframe.RemoteFrame(correction, self.db[name])

    def __getitem__(self, key):
        if isinstance(key, tuple) and key[0] in self.schemas:
            return self.get_df(key[0])[key[1:]]
        
        if key in self.schemas:
            return self.get_df(key)
        raise KeyError(key)
        
    def __dir__(self):
        return super().__dir__() + list(self.schemas)
    
    def __getattr__(self, name):            
        if name in self.schemas:
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


cframes = CorrectionFrames.default()


@export
def cmt2(name, version=0, **kwargs):
    dtime = extract_time(kwargs)
    return cframes[name].sel(time=dtime, version=version, **kwargs)

