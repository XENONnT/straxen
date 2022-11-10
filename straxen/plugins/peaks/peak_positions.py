import strax
import straxen

import numpy as np
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO

export, __all__ = strax.exporter()


@export
class PeakPositionsNT(strax.MergeOnlyPlugin):
    """
    Merge the reconstructed algorithms of the different algorithms
    into a single one that can be used in Event Basics.

    Select one of the plugins to provide the 'x' and 'y' to be used
    further down the chain. Since we already have the information
    needed here, there is no need to wait until events to make the
    decision.

    Since the computation is trivial as it only combined the three
    input plugins, don't save this plugins output.
    """
    provides = "peak_positions"
    depends_on = ("peak_positions_cnn", "peak_positions_mlp", "peak_positions_gcn")
    save_when = strax.SaveWhen.NEVER
    __version__ = '0.0.0'

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO,
        help="default reconstruction algorithm that provides (x,y)"
    )

    def infer_dtype(self):
        dtype = strax.merged_dtype([self.deps[d].dtype_for(d) for d in self.depends_on])
        dtype += [('x', np.float32, 'Reconstructed S2 X position (cm), uncorrected'),
                  ('y', np.float32, 'Reconstructed S2 Y position (cm), uncorrected')]
        return dtype

    def compute(self, peaks):
        result = {dtype: peaks[dtype] for dtype in peaks.dtype.names}
        algorithm = self.default_reconstruction_algorithm
        for xy in ('x', 'y'):
            result[xy] = peaks[f'{xy}_{algorithm}']
        return result
