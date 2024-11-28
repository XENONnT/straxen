import strax
from ._peak_positions_base import PeakPositionsBase

export, __all__ = strax.exporter()


@export
class PeakPositionsMLP(PeakPositionsBase):

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "mlp"

    tf_model_mlp = straxen.URLConfig(
        default=(
            "tf://"
            "resource://"
            "xedocs://posrec_models"
            "?attr=value"
            "&fmt=abs_path"
            "&kind=mlp"
            "&run_id=plugin.run_id"
            "&version=ONLINE"
        ),
        help='MLP model. Should be opened using the "tf" descriptor. '
        'Set to "None" to skip computation',
        cache=3,
    )
