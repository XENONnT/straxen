import strax
import numpy as np
import numba
import straxen


export, __all__ = strax.exporter()

@export
class BiPoVariables(strax.LoopPlugin):
    """
    Compute:
    - peak properties
    - peak positions
    of the first three main (in area) S1 and ten S2.
    
    The standard PosRec algorithm and the three different PosRec algorithms (mlp, gcn, cnn)
    are given for the five S2.
    """
        
    __version__ = '3.0.0'
    
    depends_on = ('events',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')
    
    # TODO change name
    provides = 'bi_po_variables'
    data_kind = 'events'
    loop_over = 'events'
    
    max_n_s1 = straxen.URLConfig(default=3, infer_type=False,
                                    help='Number of S1s to consider')

    max_n_s2 = straxen.URLConfig(default=10, infer_type=False,
                                    help='Number of S2s to consider')

    def setup(self):

        self.peak_properties = (
        # name                dtype       comment
        ('time',              np.int64,   'start time since unix epoch [ns]'),
        ('center_time',       np.int64,   'weighted center time since unix epoch [ns]'),
        ('endtime',           np.int64,   'end time since unix epoch [ns]'),
        ('area',              np.float32, 'area, uncorrected [PE]'),
        ('n_channels',        np.int32,   'count of contributing PMTs'),
        ('n_competing',       np.float32, 'number of competing PMTs'),
        ('max_pmt',           np.int16,   'PMT number which contributes the most PE'),
        ('max_pmt_area',      np.float32, 'area in the largest-contributing PMT (PE)'),
        ('range_50p_area',    np.float32, 'width, 50% area [ns]'),
        ('range_90p_area',    np.float32, 'width, 90% area [ns]'),
        ('rise_time',         np.float32, 'time between 10% and 50% area quantiles [ns]'),
        ('area_fraction_top', np.float32, 'fraction of area seen by the top PMT array')
        )
        self.to_store = [name for name, _, _ in peak_properties]

        self.pos_rec_labels = ['cnn', 'gcn', 'mlp'] # sorted alphabetically
        self.posrec_save = [(xy + algo, xy + algo) for xy in ['x_', 'y_'] for algo in pos_rec_labels] # ???? 

    def infer_dtype(self):
                
        # Basic event properties  
        basics_dtype = []
        basics_dtype += strax.time_fields
        basics_dtype += [('n_peaks', np.int32, 'Number of peaks in the event'),
                        ('n_incl_peaks_s1', np.int32, 'Number of included S1 peaks in the event'),
                        ('n_incl_peaks_s2', np.int32, 'Number of included S2 peaks in the event')]

        # For S1s and S2s
        for p_type in [1, 2]:
            if p_type == 1:
                max_n = self.max_n_s1
            if p_type == 2:
                max_n = self.max_n_s2
            for n in range(max_n):
                # Peak properties
                for name, dt, comment in self.peak_properties:
                    basics_dtype += [(f's{p_type}_{name}_{n}', dt, f'S{p_type}_{n} {comment}'), ]                

                if p_type == 2:
                    # S2 Peak positions
                    for algo in self.pos_rec_labels:
                        basics_dtype += [(f's2_x_{algo}_{n}', 
                                          np.float32, f'S2_{n} {algo}-reconstructed X position, uncorrected [cm]'),
                                         (f's2_y_{algo}_{n}',
                                          np.float32, f'S2_{n} {algo}-reconstructed Y position, uncorrected [cm]')]

        return basics_dtype
            
    def compute_loop(self, event, peaks):
        
        result = dict(time=event['time'],
                      endtime=strax.endtime(event))
        result['n_peaks'] = len(peaks)

        if not len(peaks):
            return result   
                
        ########
        #  S1  #
        ########
        
        mask_s1s  = (peaks['type']==1) 
        mask_s1s &= (peaks['area']>100)

        if not len(peaks[mask_s1s]):
            return result
        
        ## Save the biggest peaks
        max_s1s = min(self.max_n_s1, len(peaks[mask_s1s]))        
 
        # Need to initialize them to be able to use them in S2 mask without errors
        result['s1_time_0'], result['s1_time_1'] = float('nan'), float('nan')
        for i, p in enumerate(reversed(np.sort(peaks[mask_s1s], order='area'))): 
                
            for prop in self.to_store:
                result[f's1_{prop}_{i}'] = p[prop] 
            if i == self.max_n_s1 - 1:
                break

        result['n_incl_peaks_s1'] = max_s1s

        ########
        ## S2  #
        ########
        
        # TODO
        # all this mask thingis should me moved to the next plugin 
        # you can have a minimal one but should be very basic
        # the complicated stuff should be in the matching plugin
        # and this one can be used generally (not only for bipos)
        # same for the S1s
        
        mask_s2s  = peaks['type']==2
        mask_s2s &= peaks['area'] > 1500                                # low area limit
        mask_s2s &= peaks['area_fraction_top'] > 0.5                    # to remove S1 afterpulses
        mask_s2s &= np.abs(peaks['time'] - result['s1_time_0']) > 1000  # again to remove afterpulses
        mask_s2s &= np.abs(peaks['time'] - result['s1_time_1']) > 1000  # and again to remove afterpulses
        mask_s2s &= peaks['time']-result['s1_time_0'] < 5000000         # 5000mus, S2 is too far in time, not related to Po
        
        if not len(peaks[mask_s2s]):
            return result
        
        max_s2s = min(self.max_n_s2, len(peaks[mask_s2s]))
        for i, p in enumerate(reversed(np.sort(peaks[mask_s2s], order='area'))):
            for prop in self.to_store:
                result[f's2_{prop}_{i}'] = p[prop]  
            for name_alg in self.posrec_save:
                result[f's2_{name_alg[0]}_{i}'] = p[name_alg[1]]
            if i == self.max_n_s2 - 1:
                break
        
        result['n_incl_peaks_s2'] = max_s2s        

        return result
