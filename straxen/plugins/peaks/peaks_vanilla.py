from typing import Tuple, Union
import numpy as np
import strax
import straxen

from straxen.plugins.defaults import FAR_XYPOS_S2_TYPE

export, __all__ = strax.exporter()


@export
class PeaksVanilla(strax.Plugin):
    """Merge peaklets and merged S2s such that we obtain our peaks (replacing all peaklets that were
    later re-merged as S2s).

    As this step is computationally trivial, never save this plugin.

    """

    __version__ = "0.1.2"

    depends_on: Union[Tuple[str, ...], str] = (
        "peaklets",
        "enhanced_peaklet_classification",
        "merged_s2s",
    )
    data_kind = "peaks"
    provides = "peaks"
    compressor = "zstd"
    save_when = strax.SaveWhen.EXPLICIT

    diagnose_sorting = straxen.URLConfig(
        track=False,
        default=False,
        infer_type=False,
        help="Enable runtime checks for sorting and disjointness",
    )

    merge_s0 = straxen.URLConfig(
        default=True,
        type=bool,
        help="Merge S0s into merged S2s",
    )

    def infer_dtype(self):
        # In case enhanced_peaklet_classification has more fields than peaklets,
        # we need to merge them
        peaklets_dtype = self.deps["peaklets"].dtype_for("peaklets")
        peaklet_classification_dtype = self.deps["enhanced_peaklet_classification"].dtype_for(
            "enhanced_peaklet_classification"
        )
        merged_dtype = strax.merged_dtype((peaklets_dtype, peaklet_classification_dtype))
        return merged_dtype

    def compute(self, peaklets, merged_s2s):
        _peaks = self.replace_merged(peaklets, merged_s2s, merge_s0=self.merge_s0)

        if self.diagnose_sorting:
            assert np.all(np.diff(_peaks["time"]) >= 0), "Peaks not sorted"
            to_check = _peaks["type"] == 2

            assert np.all(
                _peaks["time"][to_check][1:] >= strax.endtime(_peaks)[to_check][:-1]
            ), "Peaks not disjoint"

        peaks = np.zeros(len(_peaks), dtype=self.dtype)
        strax.copy_to_buffer(_peaks, peaks, f"_copy_requested_{self.provides[0]}_fields")
        return peaks

    @staticmethod
    def replace_merged(peaklets, merged_s2s, merge_s0=True):
        # pick out type FAR_XYPOS_S2_TYPE because they might overlap with other merged S2s
        peaklets_unmerged = peaklets["type"] == FAR_XYPOS_S2_TYPE
        # if not S0 not merged, also pick out type 0
        if merge_s0:
            # of course in all cases we will pick out type 1
            peaklets_unmerged |= peaklets["type"] == 1
        else:
            peaklets_unmerged |= np.isin(peaklets["type"], [0, 1])
        peaks = strax.replace_merged(peaklets[~peaklets_unmerged], merged_s2s)
        peaks = strax.sort_by_time(np.concatenate([peaklets[peaklets_unmerged], peaks]))
        return peaks
