import strax
from ._peak_positions_base import PeakPositionsBaseNT

export, __all__ = strax.exporter()


@export
class PeakPositionsCNF(PeakPositionsBaseNT):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "cnf"
    depends_on = (
        "peaklet_positions_cnf",
        "peaklet_classification",
        "merged_s2s",
        "merged_s2_positions_cnf",
    )
    provides = "peak_positions_cnf"
