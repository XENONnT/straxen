import strax
import straxen
from straxen.plugins.events._event_s2_positions_base import EventS2PositionBase


export, __all__ = strax.exporter()


@export
class EventS2PositionCNN(EventS2PositionBase):
    """
    CNN for position S2 reconstruction at event level
    """
    algorithm = "cnn"
    provides = "event_s2_positions_cnn"

    tf_event_model_cnn = straxen.URLConfig(
        default='tf://resource://xedocs://posrec_models?attr=value&fmt=abs_path&kind=cnn&run_id=plugin.run_id&version=ONLINE',
        help='CNN model. Should be opened using the "tf" descriptor. '
             'Set to "None" to skip computation',
        cache=3,
    )
