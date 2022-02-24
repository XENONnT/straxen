
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
    _NAME: ClassVar = ''
    _DATABASE: ClassVar = 'cmt2'
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
        return utilix.xent_collection(collection=cls._NAME, database=cls._DATABASE)

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
        right = self.time.right
        utc = corrections_settings.clock.utc

        if utc and right is not None:
            right = right.replace(tzinfo=pytz.UTC)

        if right is not None and right<cutoff:
            raise IndexError(f'Overlap with existing interval.')
            
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

@export
class CorrectionReference(TimeIntervalCorrection):
    _NAME = 'global_versions'
    
    version: str = rframe.Index()
    name: str = rframe.Index()
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    correction: str
    labels: dict

    def get_corrections(self, datasource=None, **overrides):
        labels = dict(self.labels, **overrides)
        if self.correction not in BaseCorrectionSchema._SCHEMAS:
            raise KeyError(f'Reference to undefined schema name {self.correction}')
        schema = BaseCorrectionSchema._SCHEMAS[self.correction]
        return schema.find(datasource, **labels)


@export
class ResourceReference(TimeIntervalCorrection):
    fmt: ClassVar = 'text'

    value: str

    def get_resource(self, **kwargs):
        kwargs.setdefault('fmt', self.fmt)
        return straxen.get_resource(self.value, **kwargs)

@export
class BaselineSamples(TimeIntervalCorrection):
    _NAME = "baseline_samples"
    detector: str = rframe.Index()
    value: int

@export
class ElectronDriftVelocity(TimeIntervalCorrection):
    _NAME = "electron_drift_velocities"
    value: float

@export
class ElectronLifetime(TimeIntervalCorrection):
    _NAME = "electron_lifetimes"
    value: float

@export
class FdcMapName(ResourceReference):
    _NAME = "fdc_map_names"
    fmt = 'json.gz'

    kind: Literal['cnn','gcn','mlp'] = rframe.Index() 
    value: str


@export
class ModelName(ResourceReference):
    _NAME = "model_names"
    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    value: str

@export
class HitThresholds(TimeIntervalCorrection):
    _NAME = "hit_thresholds"
    detector: str = rframe.Index()
    pmt: int = rframe.Index()
    value: int

@export
class RelExtractionEff(TimeIntervalCorrection):
    _NAME = "rel_extraction_eff"
    value: float


@export
class S1XyzMap(ResourceReference):
    _NAME = "s1_xyz_map"
    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    value: str


@export
class S2XYMap(ResourceReference):
    _NAME = "s2_xy_map"
    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    value: str
@export
class PmtGain(TimeSampledCorrection):
    _NAME = 'pmt_gains'
    
    # Here we use a simple indexer (matches on exact value)
    # to define the pmt field
    # this will add the field to all documents and enable
    # selections on the pmt number. Since this is a index field
    # versioning will be indepentent for each pmt
    
    detector: Literal['tpc', 'nveto','muveto'] = rframe.Index()
    pmt: int = rframe.Index()
    
    value: float

@export
class Bodega(BaseCorrectionSchema):
    '''Detector parameters'''
    _NAME = 'bodega'
    
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
    _NAME = 'fax_configs'
    class Config:
        smart_union = True
        
    field: str = rframe.Index()
    experiment: Literal['1t','nt','nt_design'] = rframe.Index(default='nt')
    detector: Literal['tpc', 'muon_veto', 'neutron_veto'] = rframe.Index(default='tpc')
    science_run: str = rframe.Index()
    version: str = rframe.Index(default='nt')

    value: Union[int,float,bool,str,list,dict]
    resource: str
