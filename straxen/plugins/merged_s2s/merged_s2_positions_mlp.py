import strax
from straxen.plugins.peaklets.peaklet_positions_mlp import PeakletPositionsMLP

export, __all__ = strax.exporter()


@export
class MergedS2PositionsMLP(PeakletPositionsMLP):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "mlp"
    depends_on = "merged_s2s"
    provides = "merged_s2_positions_mlp"

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
