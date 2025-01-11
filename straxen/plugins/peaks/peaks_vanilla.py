from typing import Tuple, Union
import numpy as np
import strax
import straxen

from straxen.plugins.defaults import FAKE_MERGED_S2_TYPE, WIDE_XYPOS_S2_TYPE

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

    def infer_dtype(self):
        return self.deps["peaklets"].dtype_for("peaklets")

    def compute(self, peaklets, merged_s2s):
        # Remove fake merged S2s from dirty hack, see above
        merged_s2s = merged_s2s[merged_s2s["type"] != FAKE_MERGED_S2_TYPE]

        peaks = self.replace_merged(peaklets, merged_s2s)

        if self.diagnose_sorting:
            assert np.all(np.diff(peaks["time"]) >= 0), "Peaks not sorted"
            to_check = peaks["type"] != 1

            assert np.all(
                peaks["time"][to_check][1:] >= strax.endtime(peaks)[to_check][:-1]
            ), "Peaks not disjoint"

        result = np.zeros(len(peaks), dtype=self.dtype)
        strax.copy_to_buffer(peaks, result, f"_copy_requested_{self.provides[0]}_fields")
        return result

    @staticmethod
    def replace_merged(peaklets, merged_s2s):
        peaklets_is_s1 = peaklets["type"] == 1
        peaklets_is_wi = peaklets["type"] == WIDE_XYPOS_S2_TYPE
        # pick out type WIDE_XYPOS_S2_TYPE because they might overlap with other merged S2s
        peaks = strax.replace_merged(peaklets[~peaklets_is_wi & ~peaklets_is_s1], merged_s2s)
        peaks = strax.sort_by_time(
            np.concatenate([peaklets[peaklets_is_s1 | peaklets_is_wi], peaks])
        )
        return peaks
