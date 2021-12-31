import numbers
import pandas as pd
import datetime
import straxen

def coerce_datetime(value):
    unit = 's' if isinstance(value, numbers.Number) else None
    return pd.to_datetime(value, unit=unit).to_pydatetime()

COERCERS = {
    str: str,
    int: int,
    float: float,
    datetime.datetime: coerce_datetime,
    tuple: tuple,
}