
from collections import defaultdict
from immutabledict import immutabledict
import utilix

from .clock import SimpleClock

DEFAULT_DATABASE  ='cmt2'


class CorrectionsSettings:

    cmt_database: str = DEFAULT_DATABASE

    clock = SimpleClock()

    _datasources = {}
    
    def datasource(self, name):
        if name not in self._datasources:
            self._datasources[name] =  utilix.xent_collection(collection=name,
                                                              database=self.cmt_database)
        return self._datasources[name]

corrections_settings = CorrectionsSettings()