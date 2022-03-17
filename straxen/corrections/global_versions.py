

import rframe
import strax
import straxen

from .base_references import CorrectionReference

export, __all__ = strax.exporter()


@export
class GlobalVersion(CorrectionReference):
    _NAME = 'global_versions'

    strax_version: str = rframe.Index()
    straxen_version: str = rframe.Index()

    @classmethod
    def get_global_config(cls, version,
                        datasource=None,
                        names=None,
                        extra_labels=None):
        '''Build a context config from the given global version.
        '''
        if extra_labels is None:
            extra_labels = dict(run_id='plugin.run_id')
        refs = cls.find(datasource, version=version, alias=names)
        config = {}
        for ref in refs:
            url = ref.url_config
            if extra_labels is not None:
                url = straxen.URLConfig.format_url_kwargs(url, **extra_labels)
            config[ref.alias] = url
        return config
