import strax
from straxen.plugins.merged_s2s._merged_s2s_positions_base import MergedS2sPositionsBase

export, __all__ = strax.exporter()


@export
class MergedS2sPositionsGCN(MergedS2sPositionsBase):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "gcn"
    provides = "merged_s2s_positions_gcn"
