from datetime import datetime
import pandas as pd
import typing as ty
from pydantic import BaseModel

from .utils import TypeDispatch
from typing import ClassVar


class BaseCorrectionModel(BaseModel):
    index_fields: ClassVar = ('name', 'version')
    name: str
    version: int
    value: ty.Any
    
    def get_value(self):
        return self.value
    
    @property
    def index(self):
        return {field: getattr(self, field) 
                for field in self.index_fields}

class BaseTimeCorrection(BaseCorrectionModel):
    index_fields: ClassVar = ('name', 'version', 'time')
    time: datetime

class BaseMappingCorrection(BaseCorrectionModel):
    index_fields = ('name', 'version', 'key')
    key: ty.Union[str,int]

class BaseIntervalCorrection(BaseCorrectionModel):
    index_fields: ClassVar = ('name', 'version', 'begin')

    @property
    def interval(self):
        return pd.Interval(self.begin, self.end)

class IntIntervalCorrection(BaseIntervalCorrection):
    begin: int
    end: int
    
class TimeIntervalCorrection(BaseIntervalCorrection):
    begin: datetime
    end: datetime

class OfflineGain(TimeIntervalCorrection):
    index_fields: ClassVar = ('name', 'version', 'pmt', 'begin')
    pmt: int