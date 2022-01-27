
import time
import strax
import numbers
import straxen
import datetime

import pandas as pd
from collections import Counter

from straxen.remote_dataframes.schema import InsertionError
from .settings import corrections_settings

export, __all__ = strax.exporter()


@export
class BaseCorrectionSchema(straxen.BaseSchema):
    name = ''

    _SCHEMAS = {}

    index = straxen.IntegerIndex(name='version')

    def __init_subclass__(cls) -> None:
                
        if cls.name in BaseCorrectionSchema._SCHEMAS:
            raise TypeError(f'A correction with the name {cls.name} already exists.')
        if cls.name:
            cls._SCHEMAS[cls.name] = cls

        super().__init_subclass__()
        

@export
class TimeIntervalCorrection(BaseCorrectionSchema):
    name = ''
    index = straxen.TimeIntervalIndex(name='time',
                                left_name='begin', right_name='end')
        
    def pre_update(self, db, index, old_index, old_doc):
        super().pre_update(db, index, old_index, old_doc)
        if old_index['time'][1] is not None:
            raise IndexError(f'Overlap with existing interval.')

        if index['time'][0] != old_index['time'][0]:
            raise IndexError(f'Can only change endtime of existing interval. '
                                 f'start time must be {old_index["time"][0]}')

        utc = corrections_settings.clock.utc
        cutoff = corrections_settings.clock.cutoff_datetime()

        if index['time'][1]<cutoff:
            raise IndexError(f'You can only set online interval end to after {cutoff}. '
                                 'Values before this time may have already been used for processing.')

    @classmethod
    def example_df(cls, size=5, start=time.time()-1e5, stop=time.time()+1e5):
        from hypothesis.strategies import builds, lists
        
        time = make_datetime_interval_index(start, stop, size)
        index_names = list(cls.index_names())
        dfs = []
        for v in range(size):
            docs = lists(builds(cls), min_size=size, max_size=size).example()
            docs = [d.dict() for d in docs]
            df = pd.DataFrame(docs)
            df['version'] = v
            df['time'] = time
            dfs.append(df.set_index(index_names))
            
        return pd.concat(dfs)
    
def can_extrapolate(doc):
    # only extrapolate online (version=0) values
    if doc.get('version', 1):
        return False
    now = corrections_settings.clock.current_datetime()
    utc = corrections_settings.clock.utc
    ts = pd.to_datetime(doc.get('time', now), utc=utc)
    return ts < now


@export
class TimeSampledCorrection(BaseCorrectionSchema):
    name = ''
    index = straxen.TimeInterpolatedIndex(name='time', extrapolate=can_extrapolate)

    def pre_insert(self, db, index):
        cutoff = corrections_settings.clock.cutoff_datetime()
        utc = corrections_settings.clock.utc
        ts = pd.to_datetime(index['time'], utc=utc)

        if index['version']==0:
            if ts<cutoff:
                raise InsertionError(f'Can only insert online values for time after {cutoff}.')
            now = corrections_settings.clock.current_datetime()
            now_index = dict(index, time=now)
            now_values = self.query_db(db, **now_index)

            # no documents exist yet, nothing to do
            if not now_values:
                return 
            
            # enforce existence of document with the current time, in case its being extrapolated.
            now_doc = self.parse_obj(now_values[0])
            now_doc.save(db, **now_index)

def make_datetime_index(start, stop, step='1d'):
    if isinstance(start, numbers.Number):
        start = pd.to_datetime(start, unit='s', utc=True)
    if isinstance(stop, numbers.Number):
        stop = pd.to_datetime(stop, unit='s', utc=True)
    return pd.date_range(start, stop, freq=step)


def make_datetime_interval_index(start, stop, step='1d'):
    if isinstance(start, numbers.Number):
        start = pd.to_datetime(start, unit='s', utc=True)
    if isinstance(stop, numbers.Number):
        stop = pd.to_datetime(stop, unit='s', utc=True)
    return pd.interval_range(start, stop, periods=step)


