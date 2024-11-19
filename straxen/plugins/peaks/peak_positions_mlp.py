import strax
import straxen
from straxen.plugins.peaks._peak_positions_base import (
    PeakPositionsBaseNT,
    PeakletPositionsBaseNT,
    MergedS2sPositionsBaseNT,
    MergedPeakPositionsBaseNT,
)


export, __all__ = strax.exporter()


@export
class PeakPositionsMLP(PeakPositionsBaseNT):
    """Multilayer Perceptron (MLP) neural net for position reconstruction."""

    provides: str = "peak_positions_mlp"
    algorithm = "mlp"
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


@export
class PeakletPositionsMLP(PeakletPositionsBaseNT, PeakPositionsMLP):

    algorithm = "mlp"
    provides = "peaklet_positions_mlp"
    __version__ = "0.0.0"
    child_plugin = True


@export
class MergedS2sPositionsMLP(MergedS2sPositionsBaseNT, PeakPositionsMLP):

    algorithm = "mlp"
    provides = "merged_s2s_positions_mlp"
    __version__ = "0.0.0"
    child_plugin = True


@export
class MergedPeakPositionsMLP(MergedPeakPositionsBaseNT):

    algorithm = "mlp"
    depends_on = (
        "peaklet_positions_mlp",
        "peaklet_classification",
        "merged_s2s",
        "merged_s2s_positions_mlp",
    )
    provides = "peak_positions_mlp"

    __version__ = "0.0.0"
    child_plugin = True
