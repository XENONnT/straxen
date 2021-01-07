import numpy as np
from copy import deepcopy
import os 

class DoubleWidthConverter():
    """
    This is a small helping function that converts 
    standard row of PMT charge areas to 2D pattern
    needed for CNN models. 
    It uses doublewidth transformation to get from 
    HEX placing of PMTs to standard square binning
    """
    def PMTnumuber_doublewidth(self):
        """
        This function calculates the relation between 
        PMT channel number and position on 2D "image"
        """
        coords = {}
        row_nPMT = np.array([0, 6, 9, 12, 13, 14, 15, 
                             16, 17, 16, 17, 16, 17, 16, 
                             15, 14, 13, 12, 9, 6])
        row_nPMT_cumulative = np.cumsum(row_nPMT)
        tot_rows = len(row_nPMT)-2 #
        pairs = []
        for i in range(0, 253):
            n_row = np.argwhere( (i>=row_nPMT_cumulative[0:-1])*
                                 (i<row_nPMT_cumulative[1:])  
                               )[0][0]
            x_offset = ( 2*int(np.ceil( 
                           0.5* (row_nPMT.max()  - row_nPMT[n_row+1]))) 
                         - (row_nPMT[n_row+1]+1)%2)
            i_PMT = i - row_nPMT_cumulative[n_row]
            x_coord = x_offset + 2*i_PMT 
            y_coord = tot_rows - n_row
            coords[i] = [x_coord, y_coord]
            pairs.append( (x_coord, y_coord))
        return(coords,np.array(pairs) )
    
    def __init__(self):
        self.coords,self.pairs = self.PMTnumuber_doublewidth()
        self.size = (33,19)
        
    def get_coordinates(self):
        """
        this function returns dictionary of cooridnates
        if form dict(ch: [x,y]) 
        """
        return(deepcopy(self.coords))
    
    def get_size(self):
        """ 
        return size of the 2D image 
        """
        return(deepcopy(self.size))
    
    def convert_pattern(self, inarr):
        """
        This function converts an array of PMT areas to 2D pattern
        """
        pattern = np.zeros(self.size)
        pattern[self.pairs[:,0],self.pairs[:,1] ] = np.array(inarr)
        return(pattern)
    
    def convert_multiple_patterns(self, inarr):
        """
        This function converts a 2D array of PMT arreas (for multiple events)
        """
        pattern = np.zeros((inarr.shape[0],self.size[0],self.size[1]))
        pattern[:,self.pairs[:,0],self.pairs[:,1] ] = np.array(inarr)
        return(pattern)


import strax
import straxen
export, __all__ = strax.exporter()
@export
@strax.takes_config(
    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area (PE) is less than this',
                 default=10),
    strax.Option('n_top_pmts', default=straxen.n_top_pmts,
                 help="Number of top PMTs"),
    strax.Option("s2_cnn_model_path", 
                 help="Path to the CNN model in hdf5 format. WARING, this should include the whole model file and not the weights file", 
                 default="CNN_dw_3L_cm_maxnorm_A_lin_5_2000__Z_const_-1.hdf5"
                ),
    strax.Option("store_cnn_patterns",
                 help="Store patterns as used by CNN",
                 default=True)
)
class S2CNNPosRec(strax.Plugin):
    """
    This pluging provides S2 position reconstruction for top array
    
    returns variables : 
        - x_TFS2CNN - reconstructed X position in [ cm ]
        - y_TFS2CNN - reconstructed Y position in [ cm ]
        - patterns - 2D array of normalized areas as used for CNN
    """

    depends_on = ('peaks',)
    parallel = False
    provides = "S2CNNPosRec"
    __version__ = '0.0.1'
    def infer_dtype(self):
        dtype = [('x_TFS2CNN', np.float32,
                  'Reconstructed CNN S2 X position [ cm ] '),
                 ('y_TFS2CNN', np.float32,
                  'Reconstructed CNN S2 Y position [ cm ] '),
                 ]
        if self.config['store_cnn_patterns']:
            dtype.append( (("Patterns after DW transformation and normalization as used in CNN model",
                     "patterns"), np.float, (33, 19,)) )
        dtype += strax.time_fields
        return dtype
    
    def setup(self):
        import tensorflow as tf
        keras = tf.keras
        self.s2_cnn_model_path = str(self.config['s2_cnn_model_path'])
        print("CNN S2 reco : trying to load model from : \n\t %s"%self.s2_cnn_model_path)
        if not os.path.exists(self.s2_cnn_model_path):
            print("Local file does not exist, tryin downloading the file." )
            downloader = straxen.MongoDownloader()
            if self.s2_cnn_model_path in downloader.list_files():
                self.s2_cnn_model_path = downloader.download_single(self.s2_cnn_model_path)
                print("Path to downloaded file from database : \n\t : %s"%self.s2_cnn_model_path)
            else: 
                raise RuntimeError("Model file not found (locally or in DB): %s"%self.s2_cnn_model_path) 
        self.cnn_model = keras.models.load_model(self.s2_cnn_model_path)
        self.converter = DoubleWidthConverter()
        print("====== Loaded TF CNN model =====")
        self.cnn_model.summary()
        print("====== end of model summary =====")

    def compute(self, peaks):
        result = np.empty(len(peaks), dtype=self.dtype)
        result[:] = np.nan
        result['time'], result['endtime'] = peaks['time'], strax.endtime(peaks)
        peak_mask = ((peaks['area'] > self.config['min_reconstruction_area'])*
                     (peaks['type']==2))
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result
        areas  = peaks['area_per_channel'][peak_mask,0:self.config['n_top_pmts']]
        patterns = self.converter.convert_multiple_patterns(areas)
        patterns = patterns/patterns.max(axis=(1,2))[:,None,None]
        # renormalizing since CNNs are done normalized to PMT with the largest area
        reco_pos = self.cnn_model.predict(patterns)
        if self.config['store_cnn_patterns']:
            result['patterns'][peak_mask] = patterns
        # I assume that all the networks return cm
        result['x_TFS2CNN'][peak_mask] = reco_pos[:, 0]
        result['y_TFS2CNN'][peak_mask] = reco_pos[:, 1]
        return result

@export
class S2CNNPeakPositions(strax.Plugin):
    """
    This pluging puts XY for S2 peaks as peak_positions

    returns variables :
        - x - reconstructed X position in [ cm ]
        - y - reconstructed Y position in [ cm ]
    """
    depends_on = ('S2CNNPosRec',)
    parallel = False
    provides = 'peak_positions'
    __version__ = '0.0.1'
    dtype = [('x', np.float32,
                  'Reconstructed CNN S2 X position [ cm ] '),
             ('y', np.float32,
              'Reconstructed CNN S2 Y position [ cm ] '),
                 ]
    dtype += strax.time_fields

    def compute(self, peaks):
        result = np.empty(len(peaks), dtype=self.dtype)
        result[:] = np.nan
        result['time'], result['endtime'] = peaks['time'], strax.endtime(peaks)
        result['x'] = peaks['x_TFS2CNN']
        result['y'] = peaks['y_TFS2CNN']
        return result

from time import time
export, __all__ = strax.exporter()
@export
@strax.takes_config(
    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area (PE) is less than this',
                 default=10),
    strax.Option('n_top_pmts', default=straxen.n_top_pmts,
                 help="Number of top PMTs"),
    strax.Option("s2_cnn_model_path", 
                 help="Path to the CNN model in hdf5 format. WARING, this should include the whole model file and not the weights file", 
                 default="CNN_dw_3L_cm_maxnorm_A_lin_5_2000__Z_const_-1.hdf5"
                )
)
class S2CNNEventPositionFromEventAreas(strax.Plugin):
    """
    This pluging provides S2 CNN position reconstruction events 

    returns variables : 
        - s2_x_TFS2CNN - reconstructed X position of main S2 [ cm ]
        - s2_y_TFS2CNN - reconstructed Y position of main S2 [ cm ]
        - alt_s2_x_TFS2CNN - reconstructed X position of alternative S2 [ cm ]
        - alt_s2_y_TFS2CNN - reconstructed Y position of alternative S2 [ cm ]
    """

    depends_on = ('EventAreaPerChannel')
    parallel = False
    provides = "S2CNNEventPositions"
    __version__ = '0.0.0'
    def infer_dtype(self):
        dtype = [('s2_x_TFS2CNN', np.float32,
                  'Reconstructed CNN S2 X position of main S2 [ cm ] '),
                 ('s2_y_TFS2CNN', np.float32,
                  'Reconstructed CNN S2 Y position of main S2 [ cm ] '),
                 ('alt_s2_x_TFS2CNN', np.float32,
                  'Reconstructed CNN S2 X position of alternative S2 [ cm ] '),
                 ('alt_s2_y_TFS2CNN', np.float32,
                  'Reconstructed CNN S2 Y position of alternative S2 [ cm ] '),
                 ]
        dtype += strax.time_fields
        return dtype
    
    def setup(self):
        import tensorflow as tf
        keras = tf.keras
        self.s2_cnn_model_path = str(self.config['s2_cnn_model_path'])
        print("S2 CNN reco : trying to load model from : \n\t %s"%self.s2_cnn_model_path)
        if not os.path.exists(self.s2_cnn_model_path):
            print("Local file does not exist, tryin downloading the file." )
            downloader = straxen.MongoDownloader()
            if self.s2_cnn_model_path in downloader.list_files():
                self.s2_cnn_model_path = downloader.download_single(self.s2_cnn_model_path)
                print("Path to downloaded file from database : \n\t : %s"%self.s2_cnn_model_path)
            else: 
                raise RuntimeError("Model file not found (locally or in DB): %s"%self.s2_cnn_model_path) 
        self.cnn_model = keras.models.load_model(self.s2_cnn_model_path)
        self.converter = DoubleWidthConverter()
        print("====== Loaded TF CNN model =====")
        self.cnn_model.summary()
        print("====== end of model summary =====")

    def compute(self, events):
        result = np.empty(len(events), dtype=self.dtype)
        result[:] = np.nan
        result['time'],result['endtime']=events['time'], strax.endtime(events)
        for type_ in ['s2', 'alt_s2']:
            peak_mask = (events[type_+'_area'] > self.config['min_reconstruction_area'])
            areas = events[type_+"_area_per_channel"][peak_mask,0:self.config['n_top_pmts']]
            if not np.sum(peak_mask):
                continue
            patterns = self.converter.convert_multiple_patterns(areas)
            patterns = patterns/patterns.max(axis=(1,2))[:,None,None]
            reco_pos = self.cnn_model.predict(patterns)
            # I assume that all the networks return cm
            result[type_+'_x_TFS2CNN'][peak_mask] = reco_pos[:, 0]
            result[type_+'_y_TFS2CNN'][peak_mask] = reco_pos[:, 1]
        return result
