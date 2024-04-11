import numpy as np
import strax
from straxen.plugins.peaks.peak_basics import PeakBasics

export, __all__ = strax.exporter()


@export
class PeakBasicsSOM(PeakBasics):
    """Adds SOM fields to peak basics to be propgated to event basics."""

    __version__ = "0.0.1"
    child_plugin = True

    def infer_dtype(self):
        dtype = super().infer_dtype()
        additional_fields = [
            ("som_sub_type", np.int32, "SOM subtype of the peak(let)"),
            ("straxen_type", np.int8, "Old straxen type of the peak(let)"),
            ("loc_x_som", np.int16, "x location of the peak(let) in the SOM"),
            ("loc_y_som", np.int16, "y location of the peak(let) in the SOM"),
        ]

        return dtype + additional_fields

    def compute(self, peaks):
        peak_basics = super().compute(peaks)
        fields_to_copy = ("som_sub_type", "straxen_type", "loc_x_som", "loc_y_som")
        strax.copy_to_buffer(peaks, peak_basics, "_copy_som_information", fields_to_copy)
        return peak_basics
