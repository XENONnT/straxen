
import strax
import rframe

from typing import Literal, Union

from .base_corrections import BaseCorrectionSchema

export, __all__ = strax.exporter()



@export
class FaxConfig(BaseCorrectionSchema):
    '''fax configuration values for WFSim
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
