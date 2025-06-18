import strax
from straxen.plugins.peaklets.peaklet_positions_cnf import PeakletPositionsCNF

export, __all__ = strax.exporter()


@export
class MergedS2PositionsCNF(PeakletPositionsCNF):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "cnf"
    depends_on = "merged_s2s"
    provides = "merged_s2_positions_cnf"

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
