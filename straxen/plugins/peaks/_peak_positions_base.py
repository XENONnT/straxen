import strax
from .peaks import Peaks

export, __all__ = strax.exporter()


@export
class PeakPositionsBase(Peaks):

    __version__ = "0.0.0"
    child_plugin = True
    save_when = strax.SaveWhen.ALWAYS

    def infer_dtype(self):
        return self.deps[f"peaklet_positions_{self.algorithm}"].dtype_for(
            f"peaklet_positions_{self.algorithm}"
        )

    def compute(self, peaklets, merged_s2s):
        _merged_s2s = strax.merge_arrs([merged_s2s], dtype=peaklets.dtype, replacing=True)
        return super().compute(peaklets, _merged_s2s)
