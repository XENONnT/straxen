import strax
import numpy as np


class EventwBayesClass(strax.Plugin):
    """
    Append at event level the posterior probability for an
    S1, S2, alt_S1 and alt_S2
    """
    provides = 'event_w_bayes_class'
    depends_on = ('peak_classification_bayes', 'event_basics')
    data_kind = 'events'
    __version__ = '0.0.1'

    def infer_dtype(self):

        dtype = []
        dtype += strax.time_fields
        for name in ['s1', 's2', 'alt_s1', 'alt_s2']:
            dtype += [(f'{name}_ln_prob_s1', np.float32, f'Given an {name}, s1 ln probability')]
            dtype += [(f'{name}_ln_prob_s2', np.float32, f'Given an {name}, s2 ln probability')]

        return dtype

    def compute(self, peaks, events):

        result = np.empty(len(events), dtype=self.dtype)

        for name in ['s1', 's2', 'alt_s1', 'alt_s2']:
            result[f'{name}_ln_prob_s1'] = np.nan
            result[f'{name}_ln_prob_s2'] = np.nan
            # Select peaks based on their start time
            mask = np.in1d(peaks['time'], events[f'{name}_time'])
            mask_ev = np.in1d(events[f'{name}_time'], peaks['time'])
            result[f'{name}_ln_prob_s1'][mask_ev] = peaks['ln_prob_s1'][mask]
            result[f'{name}_ln_prob_s2'][mask_ev] = peaks['ln_prob_s2'][mask]
            result['time'] = events['time']
            result['endtime'] = events['endtime']

        return result
