import strax
import straxen
from ._peaklet_positions_base import PeakletPositionsBase


export, __all__ = strax.exporter()


@export
class PeakletPositionsMLP(PeakletPositionsBase):
    """Multilayer Perceptron (MLP) neural net for position reconstruction."""

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "mlp"
    provides = "peaklet_positions_mlp"
    gc_collect_after_compute = True

    tf_model_mlp = straxen.URLConfig(
        default=(
            "keras3://"
            "resource://"
            "xedocs://posrec_models"
            "?attr=value"
            "&fmt=abs_path"
            "&kind=mlp"
            "&run_id=plugin.run_id"
            "&version=ONLINE"
        ),
        help=(
            'MLP model. Should be opened using the "keras3" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )
