import strax
from ._peak_positions_base import PeakPositionsBase

export, __all__ = strax.exporter()


@export
class PeakPositionsMLP(PeakPositionsBase):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "mlp"
    depends_on = (
        "peaklet_positions_mlp",
        "enhanced_peaklet_classification",
        "merged_s2s",
        "merged_s2_positions_mlp",
    )
    provides = "peak_positions_mlp"
