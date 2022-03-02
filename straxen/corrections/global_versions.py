

import strax
import straxen

from .base_references import CorrectionReference

export, __all__ = strax.exporter()


@export
class GlobalVersion(CorrectionReference):
    _NAME = 'global_versions'

    @classmethod
    def get_global_config(cls, version, datasource=None, names=None, extra_labels=None):
        refs = cls.find(datasource, version=version, name=names)
        config = {}
        for ref in refs:
            url = ref.url_config
            if extra_labels is not None:
                url = straxen.URLConfig.format_url_kwargs(url, **extra_labels)
            config[ref.name] = url
        return config
