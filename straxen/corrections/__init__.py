from .base_corrections import *
from .bodega import *
from .fax import *
from .pmt_gains import *
from .resource_references import *
from .tf_models import *
from .simple_corrections import *
from .global_versions import *
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

@straxen.URLConfig.register(BaseCorrectionSchema._PROTOCOL_PREFIX)
def cmt2(name, version='ONLINE', **kwargs):
    dtime = corrections_settings.extract_time(kwargs)
    return find_one(name, time=dtime, version=version, **kwargs)
