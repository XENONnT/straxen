import strax
import numpy as np

from straxen.plugins.events.event_basics import EventBasics

export, __all__ = strax.exporter()


@export
class EventBasicsSOM(EventBasics):
    """Adds SOM fields for S1 and S2 peaks to event basics."""

    __version__ = "0.0.1"
    child_plugin = True

    def _set_dtype_requirements(self):
        # Properties to store for each peak (main and alternate S1 and S2)
        # Add here SOM types:
        super()._set_dtype_requirements()
        self.peak_properties = list(self.peak_properties)
        self.peak_properties += [
            ("som_sub_type", np.int32, "SOM subtype of the peak(let)"),
            ("straxen_type", np.int8, "Old straxen type of the peak(let)"),
            ("loc_x_som", np.int16, "x location of the peak(let) in the SOM"),
            ("loc_y_som", np.int16, "y location of the peak(let) in the SOM"),
        ]
        self.peak_properties = tuple(self.peak_properties)

    def compute(self, events, peaks):
        result = super().compute(events, peaks)
        return result
