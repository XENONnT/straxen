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


@export
class EventSOMClassification(strax.Plugin):
    """Plugin which propagates S1 SOM infromation to events as long straxen.PeakletClassification is
    still used."""

    depends_on = ("event_basics", "peak_som_classifcation")
    __version__ = "0.0.1"

    provides = "event_som_classification"

    def infer_dtype(self):
        dtype_peaklets = strax.time_fields + [
            ("s1_som_sub_type", np.int32, "Main S1 SOM subtype of the peak(let)"),
            ("s1_som_type", np.int8, "Main S1 SOM type of the peak(let)"),
            ("s1_loc_x_som", np.int16, "Main S1 x location of the peak(let) in the SOM"),
            ("s1_loc_y_som", np.int16, "Main S1 y location of the peak(let) in the SOM"),
            ("alt_s1_som_sub_type", np.int32, "Alt S1 SOM subtype of the peak(let)"),
            ("alt_s1_som_type", np.int8, "Alt S1 SOM type of the peak(let)"),
            ("alt_s1_loc_x_som", np.int16, "Alt S1 x location of the peak(let) in the SOM"),
            ("alt_s1_loc_y_som", np.int16, "Alt S1 y location of the peak(let) in the SOM"),
        ]
        return dtype_peaklets

    def propagate_field_to_event(self, peaks, events, events_som):
        peaks_in_event = strax.split_by_containment(peaks, events)
        return _propagate_field_to_event(peaks_in_event, events, events_som)

    def compute(self, peaks, events):
        events_som = np.zeros(len(events), self.dtype)
        events_som[:] = -1
        events_som["time"] = events["time"]
        events_som["endtime"] = events["endtime"]

        events_som = self.propagate_field_to_event(peaks, events, events_som)

        return events_som


def _propagate_field_to_event(peaks_in_event, events, events_som):

    for p_in_e, event, event_som in zip(peaks_in_event, events, events_som):
        main_s1 = p_in_e[event["s1_index"]]
        _main_s1_exist = event["s1_index"] != -1
        if _main_s1_exist:
            assert main_s1["time"] == event["s1_time"]
            event_som["s1_som_sub_type"] = main_s1["som_sub_type"]
            event_som["s1_som_type"] = main_s1["som_type"]
            event_som["s1_loc_x_som"] = main_s1["loc_x_som"]
            event_som["s1_loc_y_som"] = main_s1["loc_y_som"]

        _alt_s1_exist = event["alt_s1_index"] != -1
        if _alt_s1_exist:
            alt_s1 = p_in_e[event["alt_s1_index"]]
            assert alt_s1["time"] == event["alt_s1_time"], (alt_s1["time"], event["alt_s1_time"])
            event_som["alt_s1_som_sub_type"] = alt_s1["som_sub_type"]
            event_som["alt_s1_som_type"] = alt_s1["som_type"]
            event_som["alt_s1_loc_x_som"] = alt_s1["loc_x_som"]
            event_som["alt_s1_loc_y_som"] = alt_s1["loc_y_som"]

    return events_som
