import numpy as np
import strax
import straxen

from straxen.plugins.defaults import FAKE_MERGED_S2_TYPE

export, __all__ = strax.exporter()


@export
class Peaks(strax.Plugin):
    """
    Merge peaklets and merged S2s such that we obtain our peaks
    (replacing all peaklets that were later re-merged as S2s). As this
    step is computationally trivial, never save this plugin.
    """
    __version__ = '0.1.2'

    depends_on = ('peaklets', 'peaklet_classification_som',  'merged_s2s')
    data_kind = 'peaks'
    provides = 'peaks'
    parallel = True
    compressor = 'zstd'
    save_when = strax.SaveWhen.EXPLICIT

    diagnose_sorting = straxen.URLConfig(
        track=False, default=False, infer_type=False,
        help="Enable runtime checks for sorting and disjointness")

    merge_without_s1 = straxen.URLConfig(
        default=True, infer_type=False,
        help="If true, S1s will be igored during the merging. "
             "It's now possible for a S1 to be inside a S2 post merging")

    def infer_dtype(self):
        return self.deps['peaklets'].dtype_for('peaklets')

    def compute(self, peaklets, merged_s2s):
        # Remove fake merged S2s from dirty hack, see above
        merged_s2s = merged_s2s[merged_s2s['type'] != FAKE_MERGED_S2_TYPE]

        if self.merge_without_s1:
            is_s1 = peaklets['type'] == 1
            peaks = strax.replace_merged(peaklets[~is_s1], merged_s2s)
            peaks = strax.sort_by_time(np.concatenate([peaklets[is_s1],
                                                       peaks]))
        else:
            peaks = strax.replace_merged(peaklets, merged_s2s)

        if self.diagnose_sorting:
            assert np.all(np.diff(peaks['time']) >= 0), "Peaks not sorted"
            if self.merge_without_s1:
                to_check = peaks['type'] != 1
            else:
                to_check = peaks['type'] != FAKE_MERGED_S2_TYPE

            assert np.all(peaks['time'][to_check][1:]
                          >= strax.endtime(peaks)[to_check][:-1]), "Peaks not disjoint"
        return peaks
