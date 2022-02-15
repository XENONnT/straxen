import strax
import straxen
import numpy as np

class EventwBayesClass(strax.Plugin):

    provides = 'event_w_bayes_class'
    depends_on = ('peaklet_classification_bayes', 'event_basics')
    data_kind = 'events'
    __version__ = '0.0.1'


    def infer_dtype(self):
        
        dtype = []
        dtype += strax.time_fields
        dtype += [('s1_s1_ln_prob', np.float32, 'Given an S1, S1 ln probability')]
        dtype += [('s1_s2_ln_prob', np.float32, 'Given an S1, S2 ln probability')]
        dtype += [('s2_s1_ln_prob', np.float32, 'Given an S2, S1 ln probability')]
        dtype += [('s2_s2_ln_prob', np.float32, 'Given an S2, S2 ln probability')]

        return dtype

    def compute(self, peaklets, events):

        result = np.empty(len(events), dtype=self.dtype) ### caution here zeros means 1 remember this is ln(Pr), so do not initiate with zeros

        s1 = np.zeros(len(events), strax.time_fields)
        s2 = np.zeros(len(events), strax.time_fields)

        # For S1 and S2, match to peak posteriors
        for peaks, name in zip([s1, s2],
                              ['s1', 's2']):
            peaks['time'] = events[f'{name}_time']
            peaks['endtime'] = events[f'{name}_endtime']
            fci = strax.fully_contained_in(peaklets, peaks)
            # -1 if no contained, see https://github.com/AxFoundation/strax/blob/109a9da0194dcf5b99992e16ba1bf598874c7e7c/strax/processing/general.py#L105
            mask = fci >= 0
            fci = fci[mask]
            result[f'{name}_s1_ln_prob'][fci] = peaklets[mask]['s1_ln_prob']
            result[f'{name}_s2_ln_prob'][fci] = peaklets[mask]['s2_ln_prob']
            result['time'] = events['time']
            result['endtime'] = events['endtime']

        # events can be made out of one peak, this is to enusure user should not look at the other prob
        no_s1 = np.where(events['s1_index']==-1)
        if no_s1:
            result[f's1_s1prob'][no_s1] = np.nan
            result[f's1_s2prob'][no_s1] = np.nan
        
        return result
