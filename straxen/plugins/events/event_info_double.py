import numpy as np
import strax

export, __all__ = strax.exporter()


@export
class EventInfoDouble(strax.MergeOnlyPlugin):
    """
    Alternate version of event_info for Kr and other double scatter
    analyses:
     - Uses a different naming convention:
       s1 -> s1_a, alt_s1 -> s1_b, and similarly for s2s;
     - Adds s1_b_distinct_channels, which can be tricky to compute
       (since it requires going back to peaks)
    """
    __version__ = '0.1.2'
    depends_on = ['event_info', 'distinct_channels']
    save_when = strax.SaveWhen.EXPLICIT

    @staticmethod
    def rename_field(orig_name):
        special_cases = {'cs1': 'cs1_a',
                         'alt_cs1': 'cs1_b',
                         'alt_s1_delay': 'ds_s1_dt',
                         'cs2': 'cs2_a',
                         'alt_cs2': 'cs2_b',
                         'alt_s2_delay': 'ds_s2_dt',
                         'cs1_wo_timecorr': 'cs1_a_wo_timecorr',
                         'alt_cs1_wo_timecorr': 'cs1_b_wo_timecorr',
                         'cs2_wo_elifecorr': 'cs2_a_wo_elifecorr',
                         'alt_cs2_wo_elifecorr': 'cs2_b_wo_elifecorr',
                         'cs2_wo_timecorr': 'cs2_a_wo_timecorr',
                         'alt_cs2_wo_timecorr': 'cs2_b_wo_timecorr',
                         'cs2_area_fraction_top': 'cs2_a_area_fraction_top',
                         'alt_cs2_area_fraction_top': 'cs2_b_area_fraction_top',
                         'cs2_bottom': 'cs2_a_bottom',
                         'alt_cs2_bottom': 'cs2_b_bottom'}
        if orig_name in special_cases:
            return special_cases[orig_name]

        name = orig_name
        for s_i in [1, 2]:
            if name.startswith(f's{s_i}'):
                name = name.replace(f's{s_i}', f's{s_i}_a')
            if name.startswith(f'alt_s{s_i}'):
                name = name.replace(f'alt_s{s_i}', f's{s_i}_b')
        return name

    def infer_dtype(self):
        self.input_dtype = (
                strax.unpack_dtype(self.deps['event_info'].dtype)
                + [strax.unpack_dtype(self.deps['distinct_channels'].dtype)[0]])
        return [
            ((comment, self.rename_field(name)), dt)
            for (comment, name), dt in self.input_dtype]

    def compute(self, events):
        result = np.zeros(len(events), dtype=self.dtype)
        for (_, name), _ in self.input_dtype:
            result[self.rename_field(name)] = events[name]
        return result
