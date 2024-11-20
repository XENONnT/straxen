import strax
from straxen.plugins.peaklets._peaklet_positions_base import PeakletPositionsBase

export, __all__ = strax.exporter()


@export
class MergedS2sPositionsBase(PeakletPositionsBase):
    """Pose-rec on merged_s2s instead of peaks."""

    __version__ = "0.0.0"
    child_plugin = True
    depends_on = "merged_s2s"

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
