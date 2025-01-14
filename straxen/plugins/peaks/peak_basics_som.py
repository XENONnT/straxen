import strax
from straxen.plugins.peaklets.peaklet_classification_som import som_additional_fields
from straxen.plugins.peaks.peak_basics_vanilla import PeakBasicsVanilla

export, __all__ = strax.exporter()


@export
class PeakBasicsSOM(PeakBasicsVanilla):
    """Adds SOM fields to peak basics to be propgated to event basics."""

    __version__ = "0.0.1"
    child_plugin = True

    def infer_dtype(self):
        dtype = super().infer_dtype()
        return dtype + som_additional_fields

    def compute(self, peaks):
        peak_basics = super().compute(peaks)
        fields_to_copy = strax.to_numpy_dtype(som_additional_fields).names
        strax.copy_to_buffer(peaks, peak_basics, "_copy_som_information", fields_to_copy)
        return peak_basics
