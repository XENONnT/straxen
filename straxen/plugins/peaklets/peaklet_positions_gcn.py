import strax
import straxen
from ._peaklet_positions_base import PeakletPositionsBase


export, __all__ = strax.exporter()


@export
class PeakletPositionsGCN(PeakletPositionsBase):
    """Graph Convolutional Network (GCN) neural net for position reconstruction."""

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "gcn"
    provides = "peaklet_positions_gcn"

    tf_model_gcn = straxen.URLConfig(
        default=(
            "tf://"
            "resource://"
            f"cmt://{algorithm}_model"
            "?version=ONLINE"
            "&run_id=plugin.run_id"
            "&fmt=abs_path"
        ),
        help=(
            'GCN model. Should be opened using the "tf" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )
