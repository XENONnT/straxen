import strax
from straxen.plugins.peaklets.peaklet_positions_cnn import PeakletPositionsCNN

export, __all__ = strax.exporter()


@export
class MergedS2PositionsCNN(PeakletPositionsCNN):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "cnn"
    depends_on = "merged_s2s"
    provides = "merged_s2_positions_cnn"

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
