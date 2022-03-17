import strax
import strax
from straxen import URLConfig

import numpy as np

export, __all__ = strax.exporter()


@export
class CMT2Example(strax.Plugin):
    """
    Plugin which merges the information of all event data_kinds into a
    single data_type.
    """

    tpc_gains = URLConfig(default='cmt2://pmt_gains'
                                  '?version=v6'
                                  '&detector=tpc'
                                  '&run_id=plugin.run_id'
                                  '&attr=value'
                                  '&sort=pmt',
                        cache=True)

    depends_on = ('event_basics',)

    save_when = strax.SaveWhen.NEVER

    provides = 'gain_stats'

    dtype = strax.time_fields + [
        ('gain_average', np.float64, 'Average tpc gains'),
        ('gain_std', np.float64, 'stdv of tpc gains'),
        ]

    __version__ = '0.0.1'


    def compute(self, events):
        gains = self.tpc_gains
        result = np.zeros(len(events), dtype=self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']
        result['gain_average'] = np.average(gains)
        result['gain_std'] = np.std(gains)
       
        return  result

