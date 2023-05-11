import strax
import straxen
from straxen.plugins.events._event_s2_positions_base import EventS2PositionBase


export, __all__ = strax.exporter()


@export
class EventS2PositionMLP(EventS2PositionBase):
    """
    MLP neural net for S2 position reconstruction at event level
    """
    algorithm = "mlp"
    provides = "event_s2_positions_mlp"

    tf_event_model_mlp = straxen.URLConfig(
        default=f'tf://'
                f'resource://'
                f'cmt://{algorithm}_model'
                f'?version=ONLINE'
                f'&run_id=plugin.run_id'
                f'&fmt=abs_path',
        help='MLP model. Should be opened using the "tf" descriptor. '
             'Set to "None" to skip computation',
        cache=3,
    )
