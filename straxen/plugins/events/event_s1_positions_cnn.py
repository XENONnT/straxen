import strax
import straxen
from straxen.plugins.events._event_s1_positions_base import EventS1PositionBase


export, __all__ = strax.exporter()


@export
class EventS1PositionCNN(EventS1PositionBase):
    """CNN for (x,y,z) position S1 reconstruction at event level."""

    algorithm = "s1_cnn"
    provides = "event_s1_positions_cnn"

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
        ),
        help=(
            's1 position 3d reconstruction cnn model. Should be opened using the "tf" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )
