
from typing import ClassVar
import pytz
import strax
import numbers
import rframe
import datetime

import pandas as pd

from rframe.schema import InsertionError
from .settings import corrections_settings

export, __all__ = strax.exporter()


@export
class BaseCorrectionSchema(rframe.BaseSchema):
    _NAME: ClassVar = ''
    _PROTOCOL_PREFIX = 'cmt2'
    _SCHEMAS = {}

    version: str = rframe.Index()
    
    def __init_subclass__(cls) -> None:
                
        if cls._NAME in BaseCorrectionSchema._SCHEMAS:
            raise TypeError(f'A correction with the name {cls._NAME} already exists.')
        if cls._NAME:
            cls._SCHEMAS[cls._NAME] = cls
            
        super().__init_subclass__()

    @classmethod
    def default_datasource(cls):
        return corrections_settings.datasource(cls._NAME)

    def pre_update(self, db, new):
        if not self.same_values(new):
            index = ', '.join([f'{k}={v}' for k,v in self.index_labels.items()])
            raise IndexError(f'Values already set for {index}')


@export
class TimeIntervalCorrection(BaseCorrectionSchema):
    _NAME = ''

    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    def pre_update(self, db, new):
        cutoff = corrections_settings.clock.cutoff_datetime()
        current_right = self.time.right
        utc = corrections_settings.clock.utc

        if utc and current_right is not None:
            current_right = current_right.replace(tzinfo=pytz.UTC)

        if new.right is None:
            raise IndexError("Cannot set end date to None")
        
        if current_right is not None and current_right<cutoff:
            raise IndexError(f'Can only modify an interval that ends after {cutoff}')
            
        if self.time.left != new.time.left:
            raise IndexError(f'Can only change endtime of existing interval. '
                                 f'start time must be {self.time.left}')
            
        if not self.same_values(new):
            raise IndexError(f'Existing interval has different values.')

        new_right = new.time.right

        if new_right is not None and new_right<cutoff:
            raise IndexError(f'You can only set interval to end after {cutoff}. '
                                 'Values before this time may have already been used for processing.')


def can_extrapolate(doc):
    # only extrapolate online (version=0) values
    if doc['version'] != 'ONLINE':
        return False
    now = corrections_settings.clock.current_datetime()
    utc = corrections_settings.clock.utc
    ts = pd.to_datetime(doc['time'], utc=utc)
    return ts < now


@export
class TimeSampledCorrection(BaseCorrectionSchema):
    _NAME = ''

    time: datetime.datetime = rframe.InterpolatingIndex(extrapolate=can_extrapolate)

    def pre_insert(self, db):

        if self.version == 'ONLINE':
            cutoff = corrections_settings.clock.cutoff_datetime()
            utc = corrections_settings.clock.utc
            time = self.time
            if utc:
                time = time.replace(tzinfo=pytz.UTC)
            if time<cutoff:
                raise InsertionError(f'Can only insert online values for time after {cutoff}.')
            now_index = self.index_labels
            now_index['time'] = corrections_settings.clock.current_datetime()
            existing = self.find(db, **now_index)
            if existing:
                new_doc = existing[0]
                new_doc.save(db)


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
