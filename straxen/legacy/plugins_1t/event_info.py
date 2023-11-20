import strax

export, __all__ = strax.exporter()


@export
class EventInfo1T(strax.MergeOnlyPlugin):
    """Plugin which merges the information of all event data_kinds into a single data_type.

    This only uses 1T data-types as several event-plugins are nT only

    """

    depends_on = (
        "event_basics",
        "event_positions",
        "corrected_areas",
        "energy_estimates",
    )
    provides = "event_info"
    save_when = strax.SaveWhen.ALWAYS
    __version__ = "0.0.1"
