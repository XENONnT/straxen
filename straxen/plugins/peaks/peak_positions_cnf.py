import strax
from ._peak_positions_base import PeakPositionsBase

export, __all__ = strax.exporter()


@export
class PeakPositionsCNF(PeakPositionsBase):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "cnf"
    depends_on = (
        "peaklet_positions_cnf",
        "enhanced_peaklet_classification",
        "merged_s2s",
        "enhanced_merged_s2_classification",
        "merged_s2_positions_cnf",
    )
    provides = "peak_positions_cnf"
