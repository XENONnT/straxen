
import time
from typing import ClassVar, Literal, Union
import pytz
import strax
import numbers
import utilix
import rframe
import datetime

import pandas as pd
from collections import Counter

from rframe.schema import InsertionError
import straxen
from .settings import corrections_settings

export, __all__ = strax.exporter()


@export
class BaseCorrectionSchema(rframe.BaseSchema):
    _name: ClassVar = ''
    _DATABASE: ClassVar = 'cmt2'
    _SCHEMAS = {}

    version: str = rframe.Index()
    
    def __init_subclass__(cls) -> None:
                
        if cls._name in BaseCorrectionSchema._SCHEMAS:
            raise TypeError(f'A correction with the name {cls._name} already exists.')
        if cls._name:
            cls._SCHEMAS[cls._name] = cls
            
        super().__init_subclass__()

    @classmethod
    def default_datasource(cls):
        return utilix.xent_collection(collection=cls._name, database=cls._DATABASE)

    def pre_update(self, db, new):
        if not self.same_values(new):
            index = ', '.join([f'{k}={v}' for k,v in self.index_labels.items()])
            raise IndexError(f'Values already set for {index}')

@export
class TimeIntervalCorrection(BaseCorrectionSchema):
    _name = ''

    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    def pre_update(self, db, new):
            
        if self.time.right is not None:
            raise IndexError(f'Overlap with existing interval.')
            
        if self.time.left != new.time.left:
            raise IndexError(f'Can only change endtime of existing interval. '
                                 f'start time must be {self.time.left}')
            
        if not self.same_values(new):
            raise IndexError(f'Existing interval has different values.')
        
        utc = corrections_settings.clock.utc
        cutoff = corrections_settings.clock.cutoff_datetime()
        right = self.time.right
        if utc:
            right = right.replace(tzinfo=pytz.UTC)
        if right<cutoff:
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
    _name = ''

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


@export
class ResourceReference(TimeIntervalCorrection):
    fmt: ClassVar = 'text'

    value: str

    def get_resource(self, **kwargs):
        kwargs.setdefault('fmt', self.fmt)
        return straxen.get_resource(self.value, **kwargs)

@export
class BaselineSamples(TimeIntervalCorrection):
    _name = "baseline_samples"
    detector: str = rframe.Index()
    value: int

@export
class ElectronDriftVelocity(TimeIntervalCorrection):
    _name = "electron_drift_velocities"
    value: float

@export
class ElectronLifetime(TimeIntervalCorrection):
    _name = "electron_lifetimes"
    value: float

@export
class FdcMapName(ResourceReference):
    _name = "fdc_map_names"
    fmt = 'json.gz'

    kind: Literal['cnn','gcn','mlp'] = rframe.Index() 
    value: str


@export
class ModelName(ResourceReference):
    _name = "model_names"
    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    value: str

@export
class HitThresholds(TimeIntervalCorrection):
    _name = "hit_thresholds"
    detector: str = rframe.Index()
    pmt: int = rframe.Index()
    value: int

@export
class RelExtractionEff(TimeIntervalCorrection):
    _name = "rel_extraction_eff"
    value: float


@export
class S1XyzMap(ResourceReference):
    _name = "s1_xyz_map"
    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    value: str


@export
class S2XYMap(ResourceReference):
    _name = "s2_xy_map"
    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    value: str
@export
class PmtGain(TimeSampledCorrection):
    _name = 'pmt_gains'
    
    # Here we use a simple indexer (matches on exact value)
    # to define the pmt field
    # this will add the field to all documents and enable
    # selections on the pmt number. Since this is a index field
    # versioning will be indepentent for each pmt
    pmt: int = rframe.Index()
    detector: Literal['tpc', 'nveto','muveto'] = rframe.Index()
    
    value: float

@export
class Bodega(BaseCorrectionSchema):
    '''Detector parameters'''
    _name = 'bodega'
    
    field: str = rframe.Index()

    value: float
    uncertainty: float
    definition: str
    reference: str
    date: datetime.datetime


@export
class FaxConfig(BaseCorrectionSchema):
    '''fax configuration values
    '''
    _name = 'fax_configs'
    class Config:
        smart_union = True
        
    field: str = rframe.Index()
    experiment: Literal['1t','nt','nt_design'] = rframe.Index(default='nt')
    detector: Literal['tpc', 'muon_veto', 'neutron_veto'] = rframe.Index(default='tpc')
    science_run: str = rframe.Index()
    version: str = rframe.Index(default='nt')

    value: Union[int,float,bool,str,list,dict]
    resource: str
