import strax
from straxen.plugins.defaults import HE_PREAMBLE
from straxen.plugins.peaks.peak_basics_vanilla import PeakBasicsVanilla


export, __all__ = strax.exporter()


@export
class PeakBasicsHighEnergy(PeakBasicsVanilla):
    __doc__ = HE_PREAMBLE + (PeakBasicsVanilla.__doc__ or "")

    __version__ = "0.0.2"
    depends_on = "peaks_he"
    provides = "peak_basics_he"
    child_ends_with = "_he"

    def compute(self, peaks_he):
        return super().compute(peaks_he)
