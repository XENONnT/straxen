import strax
import straxen
from straxen.plugins.peaks._peak_positions_base import PeakPositionsBase


export, __all__ = strax.exporter()


@export
class PeakPositionsCNN(PeakPositionsBase):
    """Convolutional Neural Network (CNN) neural net for position reconstruction."""

    provides = "peak_positions_cnn"
    algorithm = "cnn"
    __version__ = "0.0.1"

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
