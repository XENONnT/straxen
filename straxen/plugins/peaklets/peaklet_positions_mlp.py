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
            "tf://"
            "resource://"
            "xnt_mlp_SR0_mix_2000031_2000021_20211211.keras"
            "?readable=True"
            "&fmt=abs_path"
        ),
        help=(
            'MLP model. Should be opened using the "tf" descriptor. '
            'Set to "None" to skip computation'
        ),
        cache=3,
    )
