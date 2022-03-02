

import strax
import straxen
import rframe
import datetime
from typing import ClassVar

from .base_corrections import BaseCorrectionSchema

export, __all__ = strax.exporter()



from .base_corrections import TimeIntervalCorrection


@export
class CorrectionReference(TimeIntervalCorrection):
    '''A CorrectionReference document references one or 
    more corrections by storing the name and labels required
    to locate the correction in a datasource     
    '''
    _NAME = ''

    name: str = rframe.Index()
    version: str = rframe.Index()
    time: rframe.Interval[datetime.datetime] = rframe.IntervalIndex()

    correction: str
    labels: dict

    def load(self, datasource=None, **overrides):
        labels = dict(self.labels, **overrides)
        if self.correction not in BaseCorrectionSchema._SCHEMAS:
            raise KeyError(f'Reference to undefined schema name {self.correction}')
        schema = BaseCorrectionSchema._SCHEMAS[self.correction]
        return schema.find(datasource, **labels)

    @property
    def url_config(self):
        url = f'{self._PROTOCOL_PREFIX}://{self.correction}'
        url = straxen.URLConfig.format_url_kwargs(url, run_id='plugin.run_id', **self.labels)
        return url

    @property
    def config_dict(self):
        return {self.name: self.url_config}

@export
class ResourceReference(BaseCorrectionSchema):
    fmt: ClassVar = 'text'

    value: str

    def pre_insert(self, db):
        self.load()
        super().pre_insert(db)

    def load(self, **kwargs):
        kwargs.setdefault('fmt', self.fmt)
        return straxen.get_resource(self.value, **kwargs)

    @property
    def url_config(self):
        return f'resource://{self.value}?fmt={self.fmt}'
