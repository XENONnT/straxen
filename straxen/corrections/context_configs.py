from typing import Dict
import rframe
import strax
import straxen

from .base_references import BaseCorrectionSchema
from .plugin_lineages import PluginLineage

export, __all__ = strax.exporter()


@export
class ContextConfig(BaseCorrectionSchema):
    _NAME = 'contexts'

    strax: str = rframe.Index()
    straxen: str = rframe.Index()

    lineage_hashes: Dict[str,str]
    
    def load_config(self, datasource=None):
        hashes = list(self.lineage_hashes.values())
        docs = PluginLineage.find(datasource, lineage_hash=hashes)
        config = {}
        for doc in docs:
            config.update(doc.config)
        return config
