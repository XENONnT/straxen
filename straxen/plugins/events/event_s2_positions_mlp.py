import strax
import straxen
from straxen.plugins.events._event_s2_positions_base import EventS2PositionBase


export, __all__ = strax.exporter()


@export
class EventS2PositionMLP(EventS2PositionBase):
    """MLP neural net for S2 position reconstruction at event level."""

    algorithm = "mlp"
    provides = "event_s2_positions_mlp"

    tf_model_mlp = straxen.URLConfig(
        default=(
            "tf://"
            "resource://"
            "xnt_mlp_SR0_mix_2000031_2000021_20211211.keras"
            "?fmt=abs_path"
            "&readable=True"
        ),
        help=(
            'MLP model. Should be opened using the "tf" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )
