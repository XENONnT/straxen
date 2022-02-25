import strax
import straxen
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
            container = np.zeros(len(events), strax.time_fields)
            container['time'] = events[f'{name}_time']
            container['endtime'] = events[f'{name}_endtime']

            mask = strax.touching_windows(peaks, container)
            result[f'{name}_ln_prob_s1'] = peaks['ln_prob_s1'][mask[:,0]]
            result[f'{name}_ln_prob_s2'] = peaks['ln_prob_s2'][mask[:,0]]
        result['time'] =  events['time']
        result['endtime'] =  events['endtime']

        # events can be made out of one peak, this is to enusure user should not look at prob
        for name in (['s1', 'alt_s1', 'alt_s2']):
            no_s_peak = np.where(events[f'{name}_index'] == -1)
            if no_s_peak:
                result[f'{name}_ln_prob_s1'][no_s_peak] = np.nan
                result[f'{name}_ln_prob_s2'][no_s_peak] = np.nan 
        
        return result
