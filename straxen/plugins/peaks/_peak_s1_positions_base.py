"""S1 Position reconstruction for Xenon-nT
Author and maintainer: Matteo Guida (guidam@mpi-hd.mpg.de)"""

import strax
import straxen
import numpy as np
from warnings import warn
import time
export, __all__ = strax.exporter()

@export
class PeakS1PositionBase(strax.Plugin):
    """
    Base pluging for S1 position reconstruction at peak level.
    Provides x_algorithm, y_algorithm, z_algorithm for all peaks > than
    min-reconstruction area based on both the top and bottom array.
    """
    __version__ = '0.0.0'
    depends_on = ('peaks',)

    algorithm = None
    compressor = 'zstd'
    parallel = True  # can set to "process" after #82

    min_s1_area_s1_posrec = straxen.URLConfig(
        help='Skip reconstruction if area (PE) is less than this',
        default=1000, infer_type=False)

    def infer_dtype(self):
        if self.algorithm is None:
            raise NotImplementedError(f'Base class should not be used without '
                                      f'algorithm as done in {__class__.__name__}')

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
    
    def get_tf_model(self):
        """
        Simple wrapper to have several tf_peak_model_s1_cnn, ..
        point to this same function in the compute method
        """
        model = getattr(self, f'tf_peak_model_{self.algorithm}', None)
        if model is None:
            warn(f'Setting model to None for {self.__class__.__name__} will '
                 f'set only nans as output for {self.algorithm}')
        if isinstance(model, str):
            raise ValueError(f'open files from tf:// protocol! Got {model} '
                             f'instead, see tests/test_s1_posrec.py for examples.')
        return model

    def compute(self, peaks):
        result = np.ones(len(peaks), dtype=self.dtype)
        result['time'], result['endtime'] = peaks['time'], strax.endtime(peaks)

        result['x_' + self.algorithm] *= float('nan')
        result['y_' + self.algorithm] *= float('nan')
        result['z_' + self.algorithm] *= float('nan')

        model = self.get_tf_model()

        # Reconstruct position only for large peaks, otherwise severe inaccuracy.
        peak_mask = peaks['area_per_channel'].sum(axis=1) > self.min_s1_area_s1_posrec
        peak_mask &= peaks['type'] < 2

        if not np.sum(peak_mask):
            # No peaks fulfilling the conditions, return nan array.
            return result

        _in = peaks['area_per_channel'][peak_mask]

        with np.errstate(divide='ignore', invalid='ignore'):
        # Normalise patters by dividing by largest PMT output between the two arrays. 
            _in = _in / _in.max(axis=1, keepdims=True)

        # Getting actual position reconstruction
        _out = model.predict(_in)

        # writing output to the result
        result['x_' + self.algorithm][peak_mask] = _out[:, 0]
        result['y_' + self.algorithm][peak_mask] = _out[:, 1]
        result['z_' + self.algorithm][peak_mask] = _out[:, 2]    
        return result
