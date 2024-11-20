import strax
from straxen.plugins.peaklets.peaklet_positions_gcn import PeakletPositionsGCN

export, __all__ = strax.exporter()


@export
class MergedS2sPositionsGCN(PeakletPositionsGCN):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "gcn"
    depends_on = "merged_s2s"
    provides = "merged_s2s_positions_gcn"

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
