import strax
import straxen
from ._peaklet_positions_base import PeakletPositionsBaseNT


export, __all__ = strax.exporter()


@export
class PeakletPositionsMLP(PeakletPositionsBaseNT):
    """Multilayer Perceptron (MLP) neural net for position reconstruction."""

    __version__ = "0.0.0"
    child_plugin = True
    algorithm = "mlp"
    provides = "peaklet_positions_mlp"
    gc_collect_after_compute = True

    tf_model_mlp = straxen.URLConfig(
        default=(
            "tf://"
            "resource://"
            f"cmt://{algorithm}_model"
            "?version=ONLINE"
            "&run_id=plugin.run_id"
            "&fmt=abs_path"
        ),
        help=(
            'MLP model. Should be opened using the "tf" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )
