""" 
   Author and Maintainer: Matteo Guida (guidam@mpi-hd.mpg.de).
   Purpose: Novel S1-based 3D position reconstruction for Xenon-nT, based on networks trained on data top and bottom S1 PMT hit patterns, considering as "true" position the S2-based reconstructed and corrected positions. In fact, the S1-based approach cannot exceed a resolution of a couple of cm and this allows to train the networks in a data-driven way. For more indormation, please see (https://zenodo.org/record/6347581#.Yt5jeHZBy3A).
"""



import os
import tempfile
import tarfile
import numpy as np
import strax
import straxen
from warnings import warn
export, __all__ = strax.exporter()

# Current default algorithm: data-driven CNN. 
DEFAULT_S1POSREC_ALGO_OPTION = tuple([strax.Option("default_s1reconstruction_algorithm",
                 help="default S1 reconstruction algorithm that provides (x,y,z)",
                 default="cnn_s1",
                 )])

@export
@strax.takes_config(
    strax.Option("s1_cnn_model_path", 
                     help="Path to the S1 CNN model", 
                     default=("/project2/lgrandi/guidam/CNN_S1_XYZ_SAVED_MODELS/version_datadriven_00_080921")
                    ),
    strax.Option('min_s1_area_s1_posrec',
                 help='Skip reconstruction if area (PE) is less than this',
                 default=1000),
)


class S1EventPositionBase(strax.Plugin):
    """
    Base class for S1 three-dimensional position reconstruction.
    This class should only be used when subclassed for the different algorithms. 
    """

    depends_on = ('events','event_area_per_channel',)
    algorithm = None
    compressor = 'zstd'
    parallel = True
    __version__ = '0.0.0'

    def infer_dtype(self):
        if self.algorithm is None:
            raise NotImplementedError(f'Base class should not be used without '
                                      f'algorithm as done in {__class__.__name__}')
        dtype = [('x_' + self.algorithm, np.float32,
                  f'Reconstructed {self.algorithm} S1 X position (cm), uncorrected'),
                 ('y_' + self.algorithm, np.float32,
                  f'Reconstructed {self.algorithm} S1 Y position (cm), uncorrected'),
                 ('z_' + self.algorithm, np.float32,
                  f'Reconstructed {self.algorithm} S1 Z position (cm), uncorrected')
                  ]

        dtype += strax.time_fields
        return dtype

    def setup(self):
        self.model_file = self._get_model_file_name()
        if self.model_file is None:
            warn(f'No file provided for {self.algorithm}. Setting all values '
                 f'for {self.provides} to None.')
            # No further setup required
            return

        # Load the tensorflow model
        import tensorflow as tf
        if os.path.exists(self.model_file):
            print(f"Path is local. Loading {self.algorithm} TF model locally "
                  f"from disk.")
        else:
            downloader = straxen.MongoDownloader()
            try:
                self.model_file = downloader.download_single(self.model_file)
            except straxen.mongo_storage.CouldNotLoadError as e:
                raise RuntimeError(f'Model files {self.model_file} is not found') from e
        with tempfile.TemporaryDirectory() as tmpdirname:
        # Allow to load the model both from folder or from file.
            try:
                tar = tarfile.open(self.model_file, mode="r:gz")
                tar.extractall(path=tmpdirname)
                self.model = tf.keras.models.load_model(tmpdirname)
            except Exception as e:
                self.model = tf.keras.models.load_model(self.model_file)


    def compute(self,events):
        result = np.ones(len(events), dtype=self.dtype)
        result['time'], result['endtime'] = events['time'], strax.endtime(events)

        result['x_' + self.algorithm] *= float('nan')
        result['y_' + self.algorithm] *= float('nan')
        result['z_' + self.algorithm] *= float('nan')

        if self.model_file is None:
            # This plugin is disabled since no model is provided
            return result

        # Reconstruct position only for large peaks, otherwise severe inaccuracy.
        event_mask = events['s1_area_per_channel'].sum(axis=1) > self.config['min_s1_area_s1_posrec']
        
        if not np.sum(event_mask):
            # No peaks fulfilling the conditions, return nan array.
            return result

        _in = events['s1_area_per_channel'][event_mask]
        
        with np.errstate(divide='ignore', invalid='ignore'):
        # Normalise patters by dividing by largest PMT output between the two arrays. 
            _in = _in / _in.max(axis=1,keepdims=True)
            
        # Getting actual position reconstruction
        _out = self.model.predict(_in)

        # writing output to the result
        
        result['x_' + self.algorithm][event_mask] = _out[:, 0]
        result['y_' + self.algorithm][event_mask] = _out[:, 1]
        result['z_' + self.algorithm][event_mask] = _out[:, 2]
        return result

    def _get_model_file_name(self):

        config_file = f'{self.algorithm}_model'
        model_from_config = self.config.get(config_file, 'No file')
        if model_from_config == 'No file':
            raise ValueError(f'{__class__.__name__} should have {config_file} '
                             f'provided as an option.')
        if isinstance(model_from_config, str) and os.path.exists(model_from_config):
            # Allow direct path specification
            return model_from_config
        if model_from_config is None:
            # Allow None to be specified (disables processing for given posrec)
            return model_from_config

        # Use CMT
        model_file = straxen.get_config_from_cmt(self.run_id, model_from_config)
        return model_file

@export
@strax.takes_config(
    strax.Option('cnn_s1_model',
                 help='Neural network model.' 
                      'If CMT, specify as (CMT_model, (cnn_s1_model, ONLINE), True)))'
                      'Set to None to skip the computation of this plugin.',
                default="/project2/lgrandi/guidam/CNN_S1_XYZ_SAVED_MODELS/version00_020521"
#                  default=("CMT_model", ('cnn_s1_model', "None"), True)
                )
)

class S1EventPositionCNN(S1EventPositionBase):
    """Convolutional Neural Network (CNN) neural net for S1 (x,y,z) position reconstruction"""
    provides = "s1_event_positions_cnn"
    algorithm = "cnn_s1"
    __version__ = '0.0.1'


@export
@strax.takes_config(
    *DEFAULT_S1POSREC_ALGO_OPTION
)
class S1EventPosition(strax.MergeOnlyPlugin):
    """
    Merge the reconstructed S1 algorithms of the different algorithms.
    
    Select one of the plugins to provide the 'x','y' and 'z' to be used 
    further down the chain. Since we already have the information
    needed here, there is no need to wait until events to make the
    decision.
        
    Since the computation is trivial as it only combined the three 
    input plugins, don't save this plugins output.
    """
    provides = "s1_event_positions"
    depends_on = ("s1_event_positions_cnn")
    save_when = strax.SaveWhen.NEVER
    __version__ = '0.0.0'

    def infer_dtype(self):
        dtype = strax.merged_dtype([self.deps[d].dtype_for(d) for d in self.depends_on])
        dtype += [('x', np.float32, 'Reconstructed S1 X position (cm), uncorrected'),
                  ('y', np.float32, 'Reconstructed S1 Y position (cm), uncorrected'),
                  ('z', np.float32, 'Reconstructed S1 Z position (cm), uncorrected')]
        return dtype

    def compute(self, events):
        result = {dtype: events[dtype] for dtype in events.dtype.names}
        algorithm = self.config['default_s1reconstruction_algorithm']
        if not 'x_' + algorithm in events.dtype.names:
            raise ValueError
        for xyz in ('x', 'y', 'z'):
            result[xyz] = events[f'{xyz}_{algorithm}']
        return result
        