import strax
import straxen
from straxen import pre_apply_function

export, __all__ = strax.exporter()


@export
class EventInfo(strax.MergeOnlyPlugin):
    """Plugin which merges the information of all event data_kinds into a single data_type."""

    depends_on = (
        "event_basics",
        "event_positions",
        "corrected_areas",
        "energy_estimates",
    )
    save_when = strax.SaveWhen.ALWAYS
    provides = "event_info"
    __version__ = "0.0.2"

    event_info_function = straxen.URLConfig(
        default="pre_apply_function",
        infer_type=False,
        help="Function that must be applied to all event_info data. Do not change.",
    )

    def compute(self, **kwargs):
        event_info = super().compute(**kwargs)
        if self.event_info_function != "disabled":
            event_info = pre_apply_function(
                event_info,
                self.run_id,
                self.provides,
                self.event_info_function,
            )
        return event_info
