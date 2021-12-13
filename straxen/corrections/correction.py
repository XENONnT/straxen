
from typing import Any
import pytz
import time
import strax
import straxen
import utilix

import datetime

import pandas as pd


export, __all__ = strax.exporter()

# editing will not be allowed for for time periods
# this many seconds into the future 
EDITING_BUFFER = 12*3600 


@export
class BaseCorrection(straxen.BaseRecord):
    index = straxen.Index(name='version', type=int)

@export
class TimeIntervalCorrection(BaseCorrection):
    index = straxen.IntervalIndex(name='time', type=datetime.datetime,
                                left_name='begin', right_name='end')
        
    def pre_insert(self, **index):
        begin = pd.to_datetime(index['begin'], utc=True)
        cutoff = pd.to_datetime(time.time()+3600, unit='s', utc=True)
        if index['version']==0 and begin<cutoff:
            raise ValueError(f'Can only insert online intervals begining at least two hours in the future.')

def can_extrapolate(index):
    # only extrapolate online (version=0) values
    if index.get('version', 1):
        return False
    now = pd.to_datetime(time.time(), unit='s', utc=True)
    ts = pd.to_datetime(index.get('time', now), utc=True)
    return ts < now
        
@export
class TimeSampledCorrection(BaseCorrection):
    index = straxen.InterpolatedIndex(name='time', type=datetime.datetime, extrapolate=can_extrapolate)
                
    def pre_insert(self, **index):
        cutoff = pd.to_datetime(time.time()+3600, unit='s', utc=True)
        ts = pd.to_datetime(index['time'], utc=True)
        if index['version']==0 and ts<cutoff:
            raise ValueError(f'Can only insert online values for times at least two hours in the future.')
