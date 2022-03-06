
import re
from typing import ClassVar
import pytz
import strax
import numbers
import rframe
import datetime
import pandas as pd

from rframe.schema import InsertionError
from .settings import corrections_settings
from ..url_config import URLConfig

export, __all__ = strax.exporter()

def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


@export
class BaseCorrectionSchema(rframe.BaseSchema):
    '''Base class for all correction schemas.
    This class ensures:
        - the _NAME attribute is always unique
        - schema includes a version index
        - changing already set values is disallowed

    '''
    _NAME: ClassVar = ''
    _SCHEMAS = {}

    version: str = rframe.Index()
    
    def __init_subclass__(cls) -> None:

        if '_NAME' not in cls.__dict__:
            cls._NAME = camel_to_snake(cls.__name__)

        # if cls._NAME in BaseCorrectionSchema._SCHEMAS:
        #     raise TypeError(f'A correction with the name {cls._NAME} already exists.')

        if cls._NAME:
            if cls._NAME not in cls._SCHEMAS:
                cls._SCHEMAS[cls._NAME] = cls
            if cls._NAME not in URLConfig._LOOKUP:
                URLConfig.register(cls._NAME)(cls.url_protocol)

        super().__init_subclass__()

    @classmethod
    def default_datasource(cls):
        return corrections_settings.datasource(cls._NAME)

    @classmethod
    def url_protocol(cls, attr, **labels):
        values = [getattr(doc, attr)  for doc in cls.find(**labels)]
        if not values:
            raise KeyError(f'No documents found for {cls._NAME} with {labels}')
        if len(values) == 1:
            return values[0]
        return values

    @property
    def name(self):
        return self._NAME

    def pre_update(self, db, new):
        if not self.same_values(new):
            index = ', '.join([f'{k}={v}' for k,v in self.index_labels.items()])
            raise IndexError(f'Values already set for {index}')



@export
class TimeIntervalCorrection(BaseCorrectionSchema):
    ''' Base class for time-interval corrections
        - Adds an Interval index of type datetime
        - Enforces rules on updating intervals:
          Can only change the right side of an interval
          if right side is None and the new right side is
          after the cutoff time (default is 2 hours after current time).
    The cutoff is set to prevent values changing after already being used
    for processing data.
    '''
    _NAME = ''

    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()
    
    @classmethod
    def url_protocol(cls, attr, **labels):
        labels['time'] = corrections_settings.extract_time(labels)
        return super().url_protocol(attr, **labels)

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
    ''' Base class for time-sampled corrections
        - Adds an interpolating index of type datetime
        - Enforces rules on inserting new data points
    Since extrapolation is allowed for ONLINE versions
    Inserting new points before the cutoff is disallowed
    This is to prevent setting values for times already
    processed using the extrapolated values.
    When inserting an ONLINE value after the cutoff, a
    new document with equal values to the extrapolated values
    is inserted at the current time to prevent the inserted document
    from affecting the interpolated values that have already been used
    for processing.
    '''
    _NAME = ''

    time: datetime.datetime = rframe.InterpolatingIndex(extrapolate=can_extrapolate)

    @classmethod
    def url_protocol(cls, attr, **labels):
        labels['time'] = corrections_settings.extract_time(labels)
        return super().url_protocol(attr, **labels)

    def pre_insert(self, db):
        # Inserting 
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

@export
def find_schema(name):
    schema = BaseCorrectionSchema._SCHEMAS.get(name, None)
    if schema is None:
        raise KeyError(f'Correction with name {name} not found.')
    return schema
    
@export
def find_corrections(name, **kwargs):
    return find_schema(name).find(**kwargs)

@export
def find_correction(name, **kwargs):
    return find_schema(name).find_one(**kwargs)


@URLConfig.register('cmt2')
def cmt2(name, version='ONLINE', **kwargs):
    dtime = corrections_settings.extract_time(kwargs)
    docs = find_corrections(name, time=dtime, version=version, **kwargs)
    if not docs:
        raise KeyError(f"No matching documents found for {name}.")
    if hasattr(docs[0], 'value'):
        docs = [d.value for d in docs]
    if len(docs) == 1:
        return docs[0]
    return docs
