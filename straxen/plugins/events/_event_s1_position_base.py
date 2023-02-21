import strax
import straxen
import numpy as np
from warnings import warn
import time
export, __all__ = strax.exporter()


@export
class EventS1PositionBase(strax.Plugin):
    """
    Base pluging for S1 position reconstruction at event level
    """
    __version__ = '0.0.0'
    depends_on = ('event_area_per_channel', 'event_basics')

    algorithm = None
    compressor = 'zstd'
    parallel = True  # can set to "process" after #82

    min_s1_area_s1_posrec = straxen.URLConfig(
        help='Skip reconstruction if area (PE) is less than this',
        default=1000, infer_type=False, )
    n_top_pmts = straxen.URLConfig(
        default=straxen.n_top_pmts, infer_type=False,
        help="Number of top PMTs")

    def infer_dtype(self):
        if self.algorithm is None:
            raise NotImplementedError(f'Base class should not be used without '
                                      f'algorithm as done in {__class__.__name__}')

    def infer_dtype(self):
        if self.algorithm is None:
            raise NotImplementedError(f'Base class should not be used without '
                                      f'algorithm as done in {__class__.__name__}')
        dtype = [('event_x_' + self.algorithm, np.float32,
                  f'Reconstructed {self.algorithm} S1 X position (cm), uncorrected'),
                 ('event_y_' + self.algorithm, np.float32,
                  f'Reconstructed {self.algorithm} S1 Y position (cm), uncorrected'),
                 ('event_z_' + self.algorithm, np.float32,
                  f'Reconstructed {self.algorithm} S1 Z position (cm), uncorrected')
                  ]

        dtype += strax.time_fields
        return dtype
    
    def get_tf_model(self):
        """
        Simple wrapper to have several tf_event_model_mlp, tf_event_model_cnn, ..
        point to this same function in the compute method
        """
        model = getattr(self, f'tf_event_model_{self.algorithm}', None)
        if model is None:
            warn(f'Setting model to None for {self.__class__.__name__} will '
                 f'set only nans as output for {self.algorithm}')
        if isinstance(model, str):
            raise ValueError(f'open files from tf:// protocol! Got {model} '
                             f'instead, see tests/test_s1_posrec.py for examples.')
        return model

    def compute(self,events):
        result = np.ones(len(events), dtype=self.dtype)
        result['time'], result['endtime'] = events['time'], strax.endtime(events)

        result['event_x_' + self.algorithm] *= float('nan')
        result['event_y_' + self.algorithm] *= float('nan')
        result['event_z_' + self.algorithm] *= float('nan')

        start_time = time.time()
        model = self.get_tf_model()
        method1_time = time.time() - start_time

        # Reconstruct position only for large peaks, otherwise severe inaccuracy.
        start_time = time.time()
        event_mask = events['s1_area_per_channel'].sum(axis=1) > self.config['min_s1_area_s1_posrec']
        method2_time = time.time() - start_time
        print("event_mask.shape:",event_mask.shape)
        
        if not np.sum(event_mask):
            # No peaks fulfilling the conditions, return nan array.
            return result
        
        start_time = time.time()
        _in = events['s1_area_per_channel'][event_mask]
        method3_time = time.time() - start_time
        print("_in.shape:",_in.shape)

        start_time = time.time()
        with np.errstate(divide='ignore', invalid='ignore'):
        # Normalise patters by dividing by largest PMT output between the two arrays. 
            _in = _in / _in.max(axis=1,keepdims=True)
        method4_time = time.time() - start_time

        # Getting actual position reconstruction
        start_time = time.time()
        _out = model.predict(_in)
        method5_time = time.time() - start_time

        # writing output to the result
        start_time = time.time()        
        result['event_x_' + self.algorithm][event_mask] = _out[:, 0]
        result['event_y_' + self.algorithm][event_mask] = _out[:, 1]
        result['event_z_' + self.algorithm][event_mask] = _out[:, 2]
        method6_time = time.time() - start_time
        print("Method 1 execution time: ", method1_time)
        print("Method 2 execution time: ", method2_time)
        print("Method 3 execution time: ", method3_time)
        print("Method 4 execution time: ", method4_time)
        print("Method 5 execution time: ", method5_time)
        print("Method 6 execution time: ", method6_time)
    
        return result
