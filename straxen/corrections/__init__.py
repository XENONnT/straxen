from .schemas import *
from .frames import *
from .settings import corrections_settings


def find(name, **kwargs):
    schema = BaseCorrectionSchema._SCHEMAS.get(name, None)
    if schema is None:
        raise KeyError(f'Correction with name {name} not found.')

    return schema.find(**kwargs)


def find_one(name, **kwargs):
    schema = BaseCorrectionSchema._SCHEMAS.get(name, None)
    if schema is None:
        raise KeyError(f'Correction with name {name} not found.')

    return schema.find_one(**kwargs)


def extract_time(kwargs):
    if 'time' in kwargs:
        return pd.to_datetime(kwargs.pop('time'), utc=True)
    if 'run_id' in kwargs:
        return run_id_to_time(kwargs.pop('run_id'))
    else:
        return None


def cmt2(name, version='ONLINE', **kwargs):
    dtime = extract_time(kwargs)
    return find_one(name, time=dtime, version=version, **kwargs)
