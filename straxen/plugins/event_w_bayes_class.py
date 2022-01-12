import strax
import straxen
import numpy as np

class EventwBayesClass(strax.Plugin):

    provides = 'event_w_bayes_class'
    depends_on = ('peaklet_classification', 'event_basics')
    data_kind = 'events'
    __version__ = '0.0.1'


    def infer_dtype(self):
        
        dtype = []
        dtype += strax.time_fields
        dtype += [('s1_s1prob', np.float32, 'Given an S1, S1 ln probability')]
        dtype += [('s1_s2prob', np.float32, 'Given an S1, S2 ln probability')]

        return dtype

    def compute(self, peaklets, events):

        result = np.zeros(len(events), dtype=self.dtype)
        s1 = np.zeros(len(events), strax.time_fields) 
        s1['time'] = events['s1_time']
        s1['endtime'] = events['s1_endtime']
        fci = strax.fully_contained_in(peaklets, s1)
        # -1 if no contained, see https://github.com/AxFoundation/strax/blob/109a9da0194dcf5b99992e16ba1bf598874c7e7c/strax/processing/general.py#L105
        mask = fci >= 0
        fci = fci[mask]
        result[fci]['s1_s1prob'] = peaklets[fci]['s1_prob']
        result[fci]['s1_s2prob'] = peaklets[fci]['s2_prob']
        result['time'] = events['time']
        result['endtime'] = events['endtime']
   
        return result
