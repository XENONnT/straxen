import strax
from straxen.plugins.peaklets.peaklet_positions_cnn import PeakletPositionsCNN

export, __all__ = strax.exporter()


@export
class MergedS2sPositionsCNN(PeakletPositionsCNN):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "cnn"
    provides = "merged_s2s_positions_cnn"

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
