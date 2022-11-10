import strax
import straxen
from straxen.plugins.peaks._peak_positions_base import PeakPositionsBaseNT


export, __all__ = strax.exporter()


@export
class PeakPositionsMLP(PeakPositionsBaseNT):
    """Multilayer Perceptron (MLP) neural net for position reconstruction"""
    provides = "peak_positions_mlp"
    algorithm = "mlp"

    tf_model_mlp = straxen.URLConfig(
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
