import strax
from straxen import pre_apply_function

import numpy as np

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        name='event_info_function',
        default='pre_apply_function', infer_type=False,
        help="Function that must be applied to all event_info data. Do not change.",
    )
)
class EventInfo(strax.MergeOnlyPlugin):
    """
    Plugin which merges the information of all event data_kinds into a
    single data_type.
    """
    depends_on = ['event_basics',
                  'event_positions',
                  'corrected_areas',
                  'energy_estimates',
                  # 'event_pattern_fit', <- this will be added soon
                  ]
    save_when = strax.SaveWhen.ALWAYS
    provides = 'event_info'
    __version__ = '0.0.2'

    def compute(self, **kwargs):
        event_info_function = self.config['event_info_function']
        event_info = super().compute(**kwargs)
        if event_info_function != 'disabled':
            event_info = pre_apply_function(event_info,
                                            self.run_id,
                                            self.provides,
                                            event_info_function,
                                            )
        return event_info


@export
class EventInfo1T(strax.MergeOnlyPlugin):
    """
    Plugin which merges the information of all event data_kinds into a
    single data_type.

    This only uses 1T data-types as several event-plugins are nT only
    """
    depends_on = ['event_basics',
                  'event_positions',
                  'corrected_areas',
                  'energy_estimates',
                  ]
    provides = 'event_info'
    save_when = strax.SaveWhen.ALWAYS
    __version__ = '0.0.1'
