import strax
import straxen
from straxen.plugins.peaks._peak_positions_base import PeakPositionsBaseNT


export, __all__ = strax.exporter()


@export
class PeakPositionsGCN(PeakPositionsBaseNT):
    """Graph Convolutional Network (GCN) neural net for position reconstruction."""

    provides = "peak_positions_gcn"
    algorithm = "gcn"
    __version__ = "0.0.1"

    tf_model_gcn = straxen.URLConfig(
        default=(
            'tf://'
            'resource://'
            'xedocs://posrec_models'
            '?attr=value'
            '&fmt=abs_path'
            '&kind=gcn'
            '&run_id=plugin.run_id'
            '&version=ONLINE'),
        help='GCN model. Should be opened using the "tf" descriptor. '
             'Set to "None" to skip computation',
        cache=3,
    )
