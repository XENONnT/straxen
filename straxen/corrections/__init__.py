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


def find_corrections(name, **kwargs):
    schema = BaseCorrectionSchema._SCHEMAS.get(name, None)
    if schema is None:
        raise KeyError(f'Correction with name {name} not found.')

    return schema.find(**kwargs)


def find_correction(name, **kwargs):
    schema = BaseCorrectionSchema._SCHEMAS.get(name, None)
    if schema is None:
        raise KeyError(f'Correction with name {name} not found.')

    return schema.find_one(**kwargs)
