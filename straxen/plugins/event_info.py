import strax
from straxen import pre_apply_function

import numpy as np

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        name='event_info_function',
        default='pre_apply_function',
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


@export
class EventInfoVetos(strax.Plugin):
    """
    Plugin which combines event_info with the tagged peaks information
    from muon- and neutron-veto.
    """
    __version__ = '0.0.0'
    depends_on = ('event_info', 'peak_veto_tags')
    provides = 'events_tagged'
    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for s_i in [1, 2]:
            for peak_type in ['', 'alt_']:
                dtype += [((f"Veto tag for {peak_type}S{s_i}: unatagged: 0, nveto: 1, mveto: 2, both: 3",
                            f"{peak_type}s{s_i}_veto_tag"), np.int8),
                          ((f"Time to closest veto interval for {peak_type}s{s_i}",
                            f"{peak_type}s{s_i}_dt_veto"), np.int64),
                          ]
        dtype += [(('Number of peaks tagged by NV/MV inside event', 'n_tagged_peaks'), np.int16)]
        return dtype

    def compute(self, events, peaks):
        split_tags = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']
        get_veto_tags(events, split_tags, result)

        return result


def get_veto_tags(events, split_tags, result):
    """
    Loops over events and tag main/alt S1/2 according to peak tag.

    :param events: Event_info data type to be tagged.
    :param split_tags: Tags split by events.
    """
    for tags_i, event_i, result_i in zip(split_tags, events, result):
        result_i['n_tagged_peaks'] = np.sum(tags_i['veto_tag'] > 0)
        for s_i in [1, 2]:
            for peak_type in ['', 'alt_']:
                if event_i[f'{peak_type}s{s_i}_index'] == -1:
                    continue

                index = event_i[f'{peak_type}s{s_i}_index']
                result_i[f'{peak_type}s{s_i}_veto_tag'] = tags_i[index]['veto_tag']
                result_i[f'{peak_type}s{s_i}_dt_veto'] = tags_i[index]['time_to_closest_veto']
