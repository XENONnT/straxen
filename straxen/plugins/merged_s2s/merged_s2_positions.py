import strax
from straxen.plugins.peaks._peak_positions_base import PeakPositionsBaseNT

export, __all__ = strax.exporter()


@export
class MergedS2Positions(PeakPositionsBaseNT):

    __version__ = "0.0.0"
    child_plugin = True
    provides = "merged_s2_positions"
    depends_on = (
        "merged_s2_positions_mlp",
        "merged_s2_positions_cnf",
    )

    def compute(self, merged_s2s):
        return super().compute(merged_s2s)
