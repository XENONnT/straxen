import strax
from ._peak_positions_base import PeakPositionsBase

export, __all__ = strax.exporter()


@export
class PeakPositionsCNN(PeakPositionsBase):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "cnn"
    depends_on = (
        "peaklet_positions_cnn",
        "peaklet_classification",
        "merged_s2s",
        "merged_s2_positions_cnn",
    )
    provides = "peak_positions_cnn"
