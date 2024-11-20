import strax
from straxen.plugins.peaks._peak_positions_base import PeakPositionsBase

export, __all__ = strax.exporter()


@export
class MergedS2sPositions(PeakPositionsBase):

    __version__ = "0.0.0"
    child_plugin = True
    provides = "merged_s2s_positions"
    depends_on = (
        "merged_s2s_positions_mlp",
        "merged_s2s_positions_cnf",
    )

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
