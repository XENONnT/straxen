from datetime import datetime
import pandas as pd
from pydantic import BaseModel


class BaseCorrection(BaseModel):
    name: str
    version: int
    value: ty.Any
    
    def get_value(self):
        return self.value

class IntervalCorrection(BaseCorrection):
    def as_interval(self):
        return pd.Interval(self.start, self.end)
        
class IntIntervalCorrection(IntervalCorrection):
    start: int
    end: int
    
class TimeIntervalCorrection(IntervalCorrection):
    start: datetime
    end: datetime
    
class IntIndexCorrection(BaseCorrection):
    index: int
