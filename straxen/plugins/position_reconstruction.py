"""Position reconstruction for Xenon-nT"""

import os
import tempfile
import tarfile
import numpy as np
import strax
import straxen
from warnings import warn
export, __all__ = strax.exporter()

DEFAULT_POSREC_ALGO_OPTION = tuple([strax.Option("default_reconstruction_algorithm",
                 help="default reconstruction algorithm that provides (x,y)",
                 default="mlp", infer_type=False,
                 )])


@export
@strax.takes_config(
    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area (PE) is less than this',
                 default=10, infer_type=False,),
    strax.Option('n_top_pmts', default=straxen.n_top_pmts, infer_type=False,
                 help="Number of top PMTs")
)
class PeakPositionsBaseNT(strax.Plugin):
    """
    Base class for reconstructions.
    This class should only be used when subclassed for the different
    algorithms. Provides x_algorithm, y_algorithm for all peaks > than
    min-reconstruction area based on the top array.
    """
    depends_on = ('peaks',)
    algorithm = None
    compressor = 'zstd'
    # Using parallel = 'process' is not allowed as we cannot Pickle
    # self.model during multiprocessing (to fix?)
    parallel = True
    __version__ = '0.0.0'

    def infer_dtype(self):
        if self.algorithm is None:
            raise NotImplementedError(f'Base class should not be used without '
                                      f'algorithm as done in {__class__.__name__}')
        dtype = [('x_' + self.algorithm, np.float32,
                  f'Reconstructed {self.algorithm} S2 X position (cm), uncorrected'),
                 ('y_' + self.algorithm, np.float32,
                  f'Reconstructed {self.algorithm} S2 Y position (cm), uncorrected')]
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
            tar = tarfile.open(self.model_file, mode="r:gz")
            tar.extractall(path=tmpdirname)
            self.model = tf.keras.models.load_model(tmpdirname)

    def compute(self, peaks):
        result = np.ones(len(peaks), dtype=self.dtype)
        result['time'], result['endtime'] = peaks['time'], strax.endtime(peaks)

        result['x_' + self.algorithm] *= float('nan')
        result['y_' + self.algorithm] *= float('nan')

        if self.model_file is None:
            # This plugin is disabled since no model is provided
            return result

        # Keep large peaks only
        peak_mask = peaks['area'] > self.config['min_reconstruction_area']
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result

        # Getting actual position reconstruction
        _in = peaks['area_per_channel'][peak_mask, 0:self.config['n_top_pmts']]
        with np.errstate(divide='ignore', invalid='ignore'):
            _in = _in / np.max(_in, axis=1).reshape(-1, 1)
        _in = _in.reshape(-1, self.config['n_top_pmts'])
        _out = self.model.predict(_in)

        # writing output to the result
        result['x_' + self.algorithm][peak_mask] = _out[:, 0]
        result['y_' + self.algorithm][peak_mask] = _out[:, 1]
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
        model_file = straxen.get_correction_from_cmt(self.run_id, model_from_config)
        return model_file


@export
@strax.takes_config(
    strax.Option('mlp_model',
                 help='Neural network model.' 
                      'If CMT, specify as (mlp_model, ONLINE, True)'
                      'Set to None to skip the computation of this plugin.',
                 default=('mlp_model', "ONLINE", True), infer_type=False,
                )
)
class PeakPositionsMLP(PeakPositionsBaseNT):
    """Multilayer Perceptron (MLP) neural net for position reconstruction"""
    provides = "peak_positions_mlp"
    algorithm = "mlp"


@export
@strax.takes_config(
    strax.Option('gcn_model',
                 help='Neural network model.' 
                      'If CMT, specify as  (gcn_model, ONLINE, True)'
                      'Set to None to skip the computation of this plugin.',
                 default=('gcn_model', "ONLINE", True), infer_type=False,
                )
)
class PeakPositionsGCN(PeakPositionsBaseNT):
    """Graph Convolutional Network (GCN) neural net for position reconstruction"""
    provides = "peak_positions_gcn"
    algorithm = "gcn"
    __version__ = '0.0.1'


@export
@strax.takes_config(
    strax.Option('cnn_model',
                 help='Neural network model.' 
                      'If CMT, specify as (cnn_model, ONLINE, True)'
                      'Set to None to skip the computation of this plugin.',
                 default=('cnn_model', "ONLINE", True), infer_type=False,
                )
)
class PeakPositionsCNN(PeakPositionsBaseNT):
    """Convolutional Neural Network (CNN) neural net for position reconstruction"""
    provides = "peak_positions_cnn"
    algorithm = "cnn"
    __version__ = '0.0.1'


@export
@strax.takes_config(
    *DEFAULT_POSREC_ALGO_OPTION
)
class PeakPositionsNT(strax.MergeOnlyPlugin):
    """
    Merge the reconstructed algorithms of the different algorithms 
    into a single one that can be used in Event Basics.
    
    Select one of the plugins to provide the 'x' and 'y' to be used 
    further down the chain. Since we already have the information
    needed here, there is no need to wait until events to make the
    decision.
    
    Since the computation is trivial as it only combined the three 
    input plugins, don't save this plugins output.
    """
    provides = "peak_positions"
    depends_on = ("peak_positions_cnn", "peak_positions_mlp", "peak_positions_gcn")
    save_when = strax.SaveWhen.NEVER
    __version__ = '0.0.0'

    def infer_dtype(self):
        dtype = strax.merged_dtype([self.deps[d].dtype_for(d) for d in self.depends_on])
        dtype += [('x', np.float32, 'Reconstructed S2 X position (cm), uncorrected'),
                  ('y', np.float32, 'Reconstructed S2 Y position (cm), uncorrected')]
        return dtype

    def compute(self, peaks):
        result = {dtype: peaks[dtype] for dtype in peaks.dtype.names}
        algorithm = self.config['default_reconstruction_algorithm']
        if not 'x_' + algorithm in peaks.dtype.names:
            raise ValueError
        for xy in ('x', 'y'):
            result[xy] = peaks[f'{xy}_{algorithm}']
        return result

    
@export
@strax.takes_config(
    strax.Option('recon_alg_included', help = 'The list of all reconstruction algorithm considered.',
                 default = ('_mlp', '_gcn', '_cnn'), infer_type=False,
                )
)
class S2ReconPosDiff(strax.Plugin):
    '''
    Plugin that provides position reconstruction difference for S2s in events, see note: 
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:shengchao:sr0:reconstruction_quality
    '''
    
    __version__ = '0.0.3'
    parallel = True
    depends_on = 'event_basics'
    provides = 's2_recon_pos_diff'
    save_when = strax.SaveWhen.EXPLICIT
    
    def infer_dtype(self):
        dtype = [
        ('s2_recon_avg_x', np.float32,
         'Mean value of x for main S2'),
        ('alt_s2_recon_avg_x', np.float32,
         'Mean value of x for alternatice S2'),
        ('s2_recon_avg_y', np.float32,
         'Mean value of y for main S2'),
        ('alt_s2_recon_avg_y', np.float32,
         'Mean value of y for alternatice S2'),
        ('s2_recon_pos_diff', np.float32,
         'Reconstructed position difference for main S2'),
        ('alt_s2_recon_pos_diff', np.float32,
         'Reconstructed position difference for alternative S2'),
    ]
        dtype += strax.time_fields
        return dtype

    def compute(self, events):
        
        result = np.zeros(len(events), dtype = self.dtype)
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        # Computing position difference
        self.compute_pos_diff(events, result)
        return result  

    def cal_avg_and_std(self, values, axis = 1):
        average = np.mean(values, axis = axis)
        std = np.std(values, axis = axis)
        return average, std

    def eval_recon(self, data, name_x_list, name_y_list):
        """
        This function reads the name list based on s2/alt_s2 and all recon algorithm registered
        Each row consists the reconstructed x/y and their average and standard deviation is calculated
        """
        x_avg, x_std = self.cal_avg_and_std(np.array(data[name_x_list].tolist())) #lazy fix to delete field name in array, otherwise np.mean will complain
        y_avg, y_std = self.cal_avg_and_std(np.array(data[name_y_list].tolist()))
        r_std = np.sqrt(x_std**2 + y_std**2)
        res = x_avg, y_avg, r_std
        return res

    def compute_pos_diff(self, events, result):
        
        alg_list = self.config['recon_alg_included']
        for peak_type in ['s2', 'alt_s2']:
            # Selecting S2s for pos diff
            # - must exist (index != -1)
            # - must have positive AFT
            # - must contain all alg info
            cur_s2_bool = (events[peak_type + '_index'] !=- 1)
            cur_s2_bool &= (events[peak_type + '_area_fraction_top'] > 0)
            for name in self.config['recon_alg_included']:
                cur_s2_bool &= ~np.isnan(events[peak_type+'_x'+name])
                cur_s2_bool &= ~np.isnan(events[peak_type+'_y'+name])
            
            # default value is nan, it will be ovewrite if the event satisfy the requirments
            result[peak_type + '_recon_pos_diff'][:] = np.nan
            result[peak_type + '_recon_avg_x'][:] = np.nan
            result[peak_type + '_recon_avg_y'][:] = np.nan
            
            if np.any(cur_s2_bool):
                name_x_list = []
                name_y_list = []
                for alg in alg_list:
                    name_x_list.append(peak_type + '_x' + alg)
                    name_y_list.append(peak_type + '_y' + alg)

                # Calculating average x,y, and position difference
                x_avg, y_avg, r_std = self.eval_recon(events[cur_s2_bool], name_x_list, name_y_list)
                result[peak_type + '_recon_pos_diff'][cur_s2_bool] = r_std
                result[peak_type + '_recon_avg_x'][cur_s2_bool] = x_avg
                result[peak_type + '_recon_avg_y'][cur_s2_bool] = y_avg
