import strax
from straxen.plugins.peaks._peak_positions_base import PeakPositionsBase

export, __all__ = strax.exporter()


@export
class PeakletPositions(PeakPositionsBase):

    __version__ = "0.0.0"
    child_plugin = True
    provides = "peaklet_positions"
    depends_on = (
        "peaklet_positions_cnf",
        "peaklet_positions_mlp",
    )

    def compute(self, peaklets):
        return super().compute(peaklets)
