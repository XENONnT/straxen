import strax
from straxen.plugins.events.event_basics_vanilla import EventBasicsVanilla
from straxen.plugins.peaklets.peaklet_classification_som import som_additional_fields

export, __all__ = strax.exporter()


@export
class EventBasicsSOM(EventBasicsVanilla):
    """Adds SOM fields for S1 and S2 peaks to event basics."""

    __version__ = "0.0.1"
    child_plugin = True

    def _set_dtype_requirements(self):
        # Properties to store for each peak (main and alternate S1 and S2)
        # Add here SOM types:
        super()._set_dtype_requirements()
        self.peak_properties += tuple(som_additional_fields)
