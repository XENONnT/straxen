import strax
import straxen
from straxen.plugins.events._event_s1_position_base import EventS1PositionBase


export, __all__ = strax.exporter()


@export
class EventS1PositionCNN(EventS1PositionBase):
    """
    CNN for (x,y,z) position S1 reconstruction at event level
    """
    algorithm = "s1_cnn"
    provides = "event_s1_position_cnn"
    # tf_event_model_s1_cnn = straxen.URLConfig.evaluate_dry(f'tf:///project2/lgrandi/guidam/CNN_S1_XYZ_SAVED_MODELS/xnt_s1_posrec_cnn_datadriven_00_080921.tar.gz')
    tf_event_model_s1_cnn = straxen.URLConfig(
        default=f'tf://'
                f'resource://'
                f'xedocs://posrec_models'
                f'?version=ONLINE'
                f'&run_id=plugin.run_id'
                f'&kind=s1_cnn'
                f'&fmt=abs_path'
                f'&attr=value',
        help='s1 position 3d reconstruction cnn model. Should be opened using the "tf" descriptor. '
             'Set to "None" to skip computation',
        cache=3,
)
