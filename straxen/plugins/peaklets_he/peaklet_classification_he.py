import strax

from straxen.plugins.peaklets.peaklet_classification import PeakletClassification
from straxen.plugins.defaults import HE_PREAMBLE

export, __all__ = strax.exporter()


@export
class PeakletClassificationHighEnergy(PeakletClassification):
    __doc__ = HE_PREAMBLE + (PeakletClassification.__doc__ or "")
    provides = "peaklet_classification_he"
    depends_on = "peaklets_he"
    __version__ = "0.0.2"
    child_plugin = True

    def compute(self, peaklets_he):
        return super().compute(peaklets_he)
