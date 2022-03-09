

import strax
import rframe
from typing import Any, Dict, Union

from .base_corrections import BaseCorrectionSchema

export, __all__ = strax.exporter()


@export
class PluginLineage(BaseCorrectionSchema):
    _NAME = 'plugin_lineages'
    class Config:
        smart_union = True
    data_type: str = rframe.Index()
    lineage_hash: str = rframe.Index()
    plugin: str = rframe.Index()
    plugin_version: str = rframe.Index()

    config: Dict[str,Union[tuple,Any]]
    depends_on: Dict[str,str]

    @classmethod
    def from_context(cls, context, data_type, version='ONLINE', run_id='0'):
        key = context.key_for(run_id, data_type)
        lineage = dict(key.lineage)
        plugin, plugin_version, config = lineage.pop(data_type)
        depends_on = {dtype: context.key_for(run_id, dtype).lineage_hash for dtype in lineage}
        plugin_config = cls(
            version=version,
            data_type=data_type,
            lineage_hash=key.lineage_hash,
            plugin=plugin,
            plugin_version=plugin_version,
            config=config,
            depends_on=depends_on,
        )
        return plugin_config

    @classmethod
    def load_config(cls, datasource=None, **labels):
        configs = {}
        docs = cls.find(datasource, **labels)
        for doc in docs:
            configs[doc.lineage_hash] = dict(doc.config)
        for doc in docs:
            missing_deps = [lhash for lhash in doc.depends_on.values() if lhash not in configs]
            if not missing_deps:
                continue
            for dep in cls.find(datasource, lineage_hash=missing_deps):
                configs[dep.lineage_hash] = dict(dep.config)
        combined = {}
        for config in configs.values():
            for k,v in config.items():
                if k in combined and v != combined[k]:
                    raise RuntimeError('Under-constrained search, '
                                       'results in inconsistent config.')
                combined[k] = v
        return combined

    def get_dependencies(self, datasource=None):
        lineages = list(self.depends_on.values())
        return self.find(datasource, lineage_hash=lineages)

    def get_full_config(self, datasource=None):
        config = {}
        for dep in self.get_dependencies(datasource):
            config.update(dep.config)
        return config