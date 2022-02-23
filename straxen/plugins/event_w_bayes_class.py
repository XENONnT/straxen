import strax
import straxen
import numpy as np


class EventwBayesClass(strax.Plugin):
    """
    Append at event level the posterior probability for an
    S1, S2, alt_S1 and alt_S2
    """
    provides = 'event_w_bayes_class'
    depends_on = ('peaklet_classification_bayes', 'event_basics')
    data_kind = 'events'
    __version__ = '0.0.1'

    def infer_dtype(self):

        dtype = []
        dtype += strax.time_fields
        for name in ['s1', 's2', 'alt_s1', 'alt_s2']:
            dtype += [(f'{name}_s1_ln_prob', np.float32, f'Given an {name}, s1 ln probability')]
            dtype += [(f'{name}_s2_ln_prob', np.float32, f'Given an {name}, s2 ln probability')]

        return dtype

    def compute(self, peaklets, events):

        result = np.empty(len(events), dtype=self.dtype)  # caution here zeros means 1 remember this is ln(Pr), so do not initiate with zeros

        s1 = np.zeros(len(events), strax.time_fields)
        s2 = np.zeros(len(events), strax.time_fields)
        s1_alt = np.zeros(len(events), strax.time_fields)
        s2_alt = np.zeros(len(events), strax.time_fields)

        # For S1 and S2, match to peak posteriors
        for peaks, name in zip([s1, s2, s1_alt, s2_alt], ['s1', 's2', 'alt_s1', 'alt_s2']):
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

        # events can be made out of one peak, this is to enusure user should not look at prob
        for name in (['s1', 'alt_s1', 'alt_s2']):
            no_s_peak = np.where(events[f'{name}_index'] == -1)
            if no_s_peak:
                result[f'{name}_s1_ln_prob'][no_s_peak] = np.nan
                result[f'{name}_s2_ln_prob'][no_s_peak] = np.nan

        return result
