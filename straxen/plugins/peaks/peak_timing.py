import strax
from straxen.plugins.peaks import Peaks

export, __all__ = strax.exporter()


@export
class PeakTiming(Peaks):
    """Merge peaklet_timing and merged S2s"""

    __version__ = '0.0.0'

    depends_on = ('peaklet_timing', 'peaklet_classification', 'merged_s2s_timing')
    provides = 'peak_timing'

    def infer_dtype(self):
        return self.deps['peaklet_timing'].dtype_for('peaklet_timing')
