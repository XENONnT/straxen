import numpy as np
from scipy.special import logsumexp

import strax
import straxen
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('bayes_cpt_config', default='/home/ahiguera-mx/test2/rap-ml-group/peak_classification/conditional_probabilities.npy',     
                 help="Bayes condition proability tables file"),
    strax.Option('bayes_bins_config', default='/home/ahiguera-mx/test2/rap-ml-group/peak_classification/discrete_parameter_bins.npy',
                 help="Bayes bins file"),

)
class BayesPeakletClassification(strax.Plugin):
    """
    Bayes Peaklet classification
    """
    provides = 'bayes_peaklet_classification'
    depends_on = ('peaklets',)
    parallel = True
    __version__ = '0.0.1'
    dtype = (strax.peak_interval_dtype
             + [('type', np.int8, 'Bayes peak classification type')]
             + [('s1_prob', np.float32, 'S1 probability')]
             + [('s2_prob', np.float32, 'S2 probability')]
            )

    # Descriptor configs
    s2_prob_threshold = straxen.URLConfig(
        default=0,
        help='S2 prob value threshold, above this value type=2'
    )
    num_nodes = straxen.URLConfig(
        default=50,
        help='Number of nodes'
    )
    classes = straxen.URLConfig(
        default=2,
        help='Number of classes'
    )

    def setup(self):

        self.class_prior = np.array([1. / self.classes for j in range(self.classes)])
        bins_model = self.config['bayes_bins_config']
        cpt_model = self.config['bayes_cpt_config']
        self.bins = np.load(bins_model)
        self.cpt = np.load(cpt_model)

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
        data[data < 0.0] = 0.0
        # can we do this faster?
        for i, p in enumerate(peaklets):
            fp = np.arange(0, num_samples+1, 1)*p['dt']
            xp = np.append([0.0], np.cumsum(data[i, :]) / np.sum(data[i, :]))
            cumsum_steps = np.interp(np.linspace(0., 1., self.num_nodes, endpoint=False), xp, fp)
            cumsum_steps = np.append(cumsum_steps, fp[-1])
            quantiles[i, :] = cumsum_steps[1:] - cumsum_steps[:-1]
            for j in range(self.num_nodes):
                waveforms[i, j] = np.sum(data[i, steps[j]:steps[j+1]])/(p['dt']*step_size)

        del data
        ###
        # Bin the waveforms and quantiles.
        ###
        waveform_bin_edges = self.bins[0, :][self.bins[0, :] > -1]
        waveform_num_bin_edges = len(waveform_bin_edges)
        quantile_bin_edges = self.bins[1, :][self.bins[1, :] > -1]
        quantile_num_bin_edges = len(quantile_bin_edges)

        waveform_values = np.digitize(waveforms, bins=waveform_bin_edges)-1
        waveform_values[waveform_values < 0] = int(0)
        waveform_values[waveform_values > int(waveform_num_bin_edges - 2)] = int(waveform_num_bin_edges - 2)

        quantile_values = np.digitize(quantiles, bins=quantile_bin_edges)-1
        quantile_values[quantile_values < 0] = int(0)
        quantile_values[quantile_values > int(quantile_num_bin_edges - 2)] = int(quantile_num_bin_edges - 2)

        values_for_inference = np.append(waveform_values, quantile_values, axis=1)

        ###
        # Inference of the binned values.
        ###
        distributions = [[] for i in range(self.num_nodes*2)]
        for i in np.arange(0, self.num_nodes, 1):
            distributions[i] = np.asarray(self.cpt[i, :waveform_num_bin_edges-1, :])
        for i in np.arange(self.num_nodes, self.num_nodes * 2, 1):
            distributions[i] = np.asarray(self.cpt[i, :quantile_num_bin_edges-1, :])

        lnposterior = np.zeros((len(peaklets), self.num_nodes*2, self.classes))
        for i in range(self.num_nodes*2):
            lnposterior[:, i, :] = np.log(distributions[i][values_for_inference[:, i], :])

        lnposterior_sumsamples = np.sum(lnposterior, axis=1)
        lnposterior_sumsamples = np.sum([lnposterior_sumsamples, np.log(self.class_prior)[np.newaxis, ...]])
        lnposterior_normed = lnposterior_sumsamples - logsumexp(lnposterior_sumsamples, axis=1)[..., np.newaxis]

        s1_prob = lnposterior_normed[:, 0]
        s2_prob = lnposterior_normed[:, 1]

        class_assignments = np.zeros(len(peaklets))
        # Probabilities to classes.
        C_S1 = s2_prob < self.s2_prob_threshold
        C_S2 = s2_prob >= self.s2_prob_threshold

        class_assignments[C_S1] = 1
        class_assignments[C_S2] = 2
        bayes_ptype = class_assignments

        return dict(type=bayes_ptype,
                    time=peaklets['time'],
                    dt=peaklets['dt'],
                    channel=-1,
                    length=peaklets['length'],
                    s1_prob=s1_prob,
                    s2_prob=s2_prob)
