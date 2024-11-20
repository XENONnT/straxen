import strax
import straxen
from ._peaklet_positions_base import PeakletPositionsBase


export, __all__ = strax.exporter()


@export
class PeakletPositionsCNN(PeakletPositionsBase):
    """Convolutional Neural Network (CNN) neural net for position reconstruction."""

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "cnn"
    provides = "peaklet_positions_cnn"

    tf_model_cnn = straxen.URLConfig(
        default=(
            "tf://"
            "resource://"
            f"cmt://{algorithm}_model"
            "?version=ONLINE"
            "&run_id=plugin.run_id"
            "&fmt=abs_path"
        ),
        help=(
            'CNN model. Should be opened using the "tf" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )
