import strax
import straxen
from straxen.plugins.events._event_s2_positions_base import EventS2PositionBase


export, __all__ = strax.exporter()


@export
class EventS2PositionGCN(EventS2PositionBase):
    """GCN net for position S2 reconstruction at event level."""

    algorithm = "gcn"
    provides = "event_s2_positions_gcn"

    tf_event_model_gcn = straxen.URLConfig(
        default=f"tf://"
        f"resource://"
        f"cmt://{algorithm}_model"
        f"?version=ONLINE"
        f"&run_id=plugin.run_id"
        f"&fmt=abs_path",
        help='MLP model. Should be opened using the "tf" descriptor. '
        'Set to "None" to skip computation',
        cache=3,
    )
