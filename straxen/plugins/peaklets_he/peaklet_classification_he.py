import strax

from straxen.plugins.peaklets.peaklet_classification_vanilla import PeakletClassificationVanilla
from straxen.plugins.defaults import HE_PREAMBLE

export, __all__ = strax.exporter()


@export
class PeakletClassificationHighEnergy(PeakletClassificationVanilla):
    __doc__ = HE_PREAMBLE + (PeakletClassificationVanilla.__doc__ or "")

    __version__ = "0.0.2"
    child_plugin = True
    depends_on = "peaklets_he"
    provides = "peaklet_classification_he"

    def compute(self, peaklets_he):
        return super().compute(peaklets_he)
