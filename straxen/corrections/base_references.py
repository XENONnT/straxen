

import strax
import straxen
import rframe
import datetime
from typing import ClassVar, Literal

from .base_corrections import BaseCorrectionSchema
from .settings import corrections_settings

export, __all__ = strax.exporter()



from .base_corrections import TimeIntervalCorrection


@export
class CorrectionReference(TimeIntervalCorrection):
    '''A CorrectionReference document references one or 
    more corrections by storing the name and labels required
    to locate the correction in a datasource     
    '''
    _NAME = ''

    # arbitrary alias for this reference,
    # this should match the straxen config name
    alias: str = rframe.Index() 

    # the global version
    version: str = rframe.Index()

    # validity interval of the document
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    # Name of the correction being referenced
    correction: str

    # The attribute in the correction being referenced e.g `value`
    attribute: str

    # The index labels being referenced, eg pmt=[1,2,3], version='v3' etc.
    labels: dict

    def load(self, datasource=None, **overrides):
        ''' Load the referenced documents from the
        given datasource.
        '''
        labels = dict(self.labels, **overrides)
        if self.correction not in BaseCorrectionSchema._SCHEMAS:
            raise KeyError(f'Reference to undefined schema name {self.correction}')
        schema = BaseCorrectionSchema._SCHEMAS[self.correction]
        return schema.find(datasource, **labels)

    @property
    def url_config(self):
        '''Convert reference to a URLConfig URL
        '''
        url = f'{self.correction}://{self.attribute}'
        url = straxen.URLConfig.format_url_kwargs(url, **self.labels)
        return url

    @property
    def config_dict(self):
        return {self.name: self.url_config}

@export
class BaseResourceReference(BaseCorrectionSchema):
    _NAME = ''

    fmt: ClassVar = 'text'

    value: str

    def pre_insert(self, datasource):
        '''require the existence of the resource
        being referenced prior to inserting a new
        document. This is to avoid typos etc.
        '''
        self.load()
        super().pre_insert(datasource)

    def load(self, **kwargs):
        kwargs.setdefault('fmt', self.fmt)
        return straxen.get_resource(self.value, **kwargs)

    @property
    def url_config(self):
        return f'resource://{self.value}?fmt={self.fmt}'


class BaseMap(BaseResourceReference):
    _NAME = ''

    kind: Literal['cnn','gcn','mlp'] = rframe.Index()
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    value: str
