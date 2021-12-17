import numpy as np
from scipy.special import logsumexp

import strax
import straxen
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('s1_risetime_area_parameters', default=(50, 80, 12), type=(list, tuple),
                 help="norm, const, tau in the empirical boundary in the risetime-area plot"),
    strax.Option('s1_risetime_aft_parameters', default=(-1, 2.6), type=(list, tuple),
                 help=("Slope and offset in exponential of emperical boundary in the rise time-AFT "
                      "plot. Specified as (slope, offset)")),
    strax.Option('s1_flatten_threshold_aft', default=(0.6, 100), type=(tuple, list),
                 help=("Threshold for AFT, above which we use a flatted boundary for rise time" 
                       "Specified values: (AFT boundary, constant rise time).")),
    strax.Option('n_top_pmts', default=straxen.n_top_pmts, type=int,
                 help="Number of top PMTs"),
    strax.Option('s1_max_rise_time_post100', default=200, type=(int, float),
                 help="Maximum S1 rise time for > 100 PE [ns]"),
    strax.Option('s1_min_coincidence', default=2, type=int,
                 help="Minimum tight coincidence necessary to make an S1"),
    strax.Option('s2_min_pmts', default=4, type=int,
                 help="Minimum number of PMTs contributing to an S2"),
    strax.Option('do_bayes', default=True,
                 help="run bayes classification method"),
    strax.Option('bayes_CPT_config', default='/home/ahiguera-mx/test2/rap-ml-group/peak_classification/conditional_probabilities.npy',     
                 help="Bayes condition proability tables file"),
    strax.Option('bayes_bins_config', default='/home/ahiguera-mx/test2/rap-ml-group/peak_classification/discrete_parameter_bins.npy',
                 help="Bayes bins"),
    strax.Option('s2_prob_threshold', default=-27,
                 help="S2 prob value, above this value type=2"),
)
class PeakletClassification(strax.Plugin):
    """Classify peaklets as unknown, S1, or S2."""
    provides = 'peaklet_classification'
    depends_on = ('peaklets',)
    parallel = True
    dtype = (strax.peak_interval_dtype
             + [('type', np.int8, 'Classification of the peak(let)')]
             + [('type_bayes', np.int8, 'Bayes peak classification type')]
             + [('s1_prob', np.float32, 'S1 ln probability' )]
             + [('s2_prob', np.float32, 'S2 ln probability' )]
             )

    __version__ = '4.0.0'

    @staticmethod
    def upper_rise_time_area_boundary(area, norm, const, tau):
        """
        Function which determines the upper boundary for the rise-time
        for a given area.
        """
        return norm*np.exp(-area/tau) + const

    @staticmethod
    def upper_rise_time_aft_boundary(aft, slope, offset, aft_boundary, flat_threshold):
        """
        Function which computes the upper rise time boundary as a function
        of area fraction top.
        """
        res = 10**(slope * aft + offset)
        res[aft >= aft_boundary] = flat_threshold
        return res

    def compute(self, peaklets):
        ptype = np.zeros(len(peaklets), dtype=np.int8)

        # Properties needed for classification:
        rise_time = -peaklets['area_decile_from_midpoint'][:, 1]
        n_channels = (peaklets['area_per_channel'] > 0).sum(axis=1)
        n_top = self.config['n_top_pmts']
        area_top = peaklets['area_per_channel'][:, :n_top].sum(axis=1)
        area_total = peaklets['area_per_channel'].sum(axis=1)
        area_fraction_top = area_top/area_total

        is_large_s1 = (peaklets['area'] >= 100)
        is_large_s1 &= (rise_time <= self.config['s1_max_rise_time_post100'])
        is_large_s1 &= peaklets['tight_coincidence_channel'] >= self.config['s1_min_coincidence']

        is_small_s1 = peaklets["area"] < 100
        is_small_s1 &= rise_time < self.upper_rise_time_area_boundary(
            peaklets["area"],
            *self.config["s1_risetime_area_parameters"],
        )

        is_small_s1 &= rise_time < self.upper_rise_time_aft_boundary(
            area_fraction_top,
            *self.config["s1_risetime_aft_parameters"],
            *self.config["s1_flatten_threshold_aft"],
        )

        is_small_s1 &= peaklets['tight_coincidence_channel'] >= self.config['s1_min_coincidence']

        ptype[is_large_s1 | is_small_s1] = 1

        is_s2 = n_channels >= self.config['s2_min_pmts']
        is_s2[is_large_s1 | is_small_s1] = False
        ptype[is_s2] = 2

        if self.config['do_bayes']:

            Bclassifier =PeakClassificationBayes(bins=self.config['bayes_bins_config'],
                                                 CPT=self.config['bayes_CPT_config']
                                                )
            lnposterior = Bclassifier.compute(peaklets)
            s1_prob = lnposterior[:,0]
            s2_prob = lnposterior[:,1] 
 
            # Probabilities to classes.
            C_S1 = lnposterior[:,1] < self.config['s2_prob_threshold']
            C_S2 = lnposterior[:,1] > self.config['s2_prob_threshold']
    
            class_assignments = np.zeros(len(peaklets))
            class_assignments[C_S1] = 1
            class_assignments[C_S2] = 2
            bayes_ptype = class_assignments

        return dict(type=ptype,
                    type_bayes=bayes_ptype,
                    time=peaklets['time'],
                    dt=peaklets['dt'],
                    channel=-1,
                    length=peaklets['length'],
                    s1_prob=s1_prob,
                    s2_prob=s2_prob)


class PeakClassificationBayes():
    """
    Peak classification based on Bayes classifier 
    Returns the ln probability of a each peaklet belonging to the S1 and S2 class.
    Uses conditional probabilities and data parameterization learned from wfsim data
    """
    __version__ = "0.0.1"

    def __init__(self, num_nodes=50, bins=None, CPT=None ):
        self.num_nodes = num_nodes
        self.classes = 2
        self.class_prior = np.array([1. /self.classes for j in range(self.classes)])
        self.bins = np.load(bins)
        self.CPT = np.load(CPT)


    def compute(self, peaklets):
        bayes_ptype = np.zeros(len(peaklets), dtype=np.int8) 
        s1_prob = np.zeros(len(peaklets), dtype=np.float32) 
        s2_prob = np.zeros(len(peaklets), dtype=np.float32)

        waveforms = np.zeros((len(peaklets), self.num_nodes))
        quantiles = np.zeros((len(peaklets), self.num_nodes))

        ###
        # calculate waveforms and quantiles.
        ###
        num_samples = peaklets['data'].shape[1]
        step_size = int(num_samples/self.num_nodes)
        steps = np.arange(0, num_samples+1, step_size)

        data = peaklets['data'].copy()
        data[data<0.0] = 0.0
        # can we do this faste?
        for i, p in enumerate(peaklets):
            fp = np.arange(0,num_samples+1,1)*p['dt'] 
            xp = np.append([0.0], np.cumsum(data[i,:])/np.sum(data[i,:]))
            cumsum_steps = np.interp(np.linspace(0.,1.,self.num_nodes, endpoint=False), xp, fp)
            cumsum_steps = np.append(cumsum_steps, fp[-1])
            quantiles[i,:] = cumsum_steps[1:]-cumsum_steps[:-1]
            for j in range(self.num_nodes):
                waveforms[i,j] = np.sum(data[i,steps[j]:steps[j+1]])/(p['dt']*step_size)

        del data
        ###
        # Bin the waveforms and quantiles.
        ###
        waveform_bin_edges = self.bins[0,:][self.bins[0,:] > -1]
        waveform_num_bin_edges = len(waveform_bin_edges)
        quantile_bin_edges = self.bins[1,:][self.bins[1,:] > -1]
        quantile_num_bin_edges = len(quantile_bin_edges)
    
    
        waveform_values = np.digitize(waveforms, bins=waveform_bin_edges)-1
        waveform_values[waveform_values<0] = int(0)
        waveform_values[waveform_values>int(waveform_num_bin_edges-2)] = int(waveform_num_bin_edges-2)
    
        quantile_values = np.digitize(quantiles, bins=quantile_bin_edges)-1
        quantile_values[quantile_values<0] = int(0)
        quantile_values[quantile_values>int(quantile_num_bin_edges-2)] = int(quantile_num_bin_edges-2)
    
        values_for_inference = np.append(waveform_values, quantile_values, axis=1)
 
        ###
        # Inference of the binned values.
        ###
    
        distributions = [[] for i in range(self.num_nodes*2)]
        for i in np.arange(0,self.num_nodes,1):
            distributions[i] = np.asarray(self.CPT[i,:waveform_num_bin_edges-1,:])
        for i in np.arange(self.num_nodes,self.num_nodes*2,1):
            distributions[i] = np.asarray(self.CPT[i,:quantile_num_bin_edges-1,:])
    
        lnposterior = np.zeros((len(peaklets), self.num_nodes*2, self.classes))
        for i in range(self.num_nodes*2):
            lnposterior[:,i,:] = np.log(distributions[i][values_for_inference[:,i],:])
    
        lnposterior_sumsamples = np.sum(lnposterior, axis=1)
    
        lnposterior_sumsamples = np.sum([lnposterior_sumsamples, np.log(self.class_prior)[np.newaxis,...]])
    
        lnposterior_normed = lnposterior_sumsamples - logsumexp(lnposterior_sumsamples, axis=1)[...,np.newaxis]

        return lnposterior_normed



