import strax
import straxen
from straxen.plugins.peaks._peak_s1_positions_base import PeakS1PositionBase


export, __all__ = strax.exporter()


@export
class PeakS1PositionCNN(PeakS1PositionBase):
    """S1 CNN for (x,y,z) position S1 reconstruction at peak level."""

    provides = "peak_s1_positions_cnn"
    algorithm = "s1_cnn"
    __version__ = "0.0.1"

    tf_model_s1_cnn = straxen.URLConfig(
        default=(
            "tf://"
            "resource://"
            "xedocs://posrec_models"
            "?version=ONLINE"
            "&run_id=plugin.run_id"
            "&kind=s1_cnn"
            "&fmt=abs_path"
            "&attr=value"
            "&readable=True"
        ),
        help=(
            's1 position 3d reconstruction cnn model. Should be opened using the "tf" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )
