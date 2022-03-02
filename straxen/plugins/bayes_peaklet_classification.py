import numpy as np
import numba
from scipy.special import logsumexp
import strax
import straxen


class BayesPeakletClassification(strax.Plugin):
    """
    Bayes Peaklet classification
    Returns the ln probability of a each event belonging to the S1 and S2 class.
    Uses conditional probabilities and data parameterization learned from wfsim data.
    More info can be found here xenon:xenonnt:ahiguera:bayespeakclassification

    :param peaks: peaks
    :param waveforms: peaks waveforms in PE/ns
    :param quantiles: quantiles in ns
    :returns: the ln probability of a each peaklet belonging to S1 and S2 class
    """
    provides = 'peak_classification_bayes'
    depends_on = ('peaks',)
    __version__ = '0.0.1'
    dtype = (strax.time_fields
             + [('ln_prob_s1', np.float32, 'S1 ln probability')]
             + [('ln_prob_s2', np.float32, 'S2 ln probability')]
            )

    # Descriptor configs
    bayes_config_file = straxen.URLConfig(
        default='resource://cmt://'
                'bayes_model'
                '?version=ONLINE&run_id=plugin.run_id&fmt=npy',
        help='Bayes model, conditional probabilities tables and Bayes discrete bins'
    )
    bayes_n_nodes = straxen.URLConfig(
        default=50,
        help='Number of attributes(features) per waveform and quantile'
    )
    n_bayes_classes = straxen.URLConfig(
        default=2,
        help='Number of label classes S1(1)/S2(2)'
    )

    def setup(self):

        self.class_prior = np.ones(self.n_bayes_classes)/self.n_bayes_classes
        self.bins = self.bayes_config_file['bins']
        self.cpt = self.bayes_config_file['cprob']

    def compute(self, peaks):

        waveforms, quantiles = compute_wf_and_quantiles(peaks, self.bayes_n_nodes)

        ln_prob_s1, ln_prob_s2 = compute_inference(self.bins, self.bayes_n_nodes, self.cpt,
                                                   self.n_bayes_classes, self.class_prior,
                                                   waveforms, quantiles)

        return dict(time=peaks['time'],
                    endtime=peaks['time'] + peaks['dt'] * peaks['length'],
                    ln_prob_s1=ln_prob_s1,
                    ln_prob_s2=ln_prob_s2
                    )

def compute_wf_and_quantiles(peaks: np.ndarray, bayes_n_nodes: int):
    """
    Compute waveforms and quantiles for a given number of nodes(atributes)
    :param peaks:
    :param bayes_n_nodes: number of nodes or atributes
    :return: waveforms and quantiles
    """
    waveforms = np.zeros((len(peaks), bayes_n_nodes))
    quantiles = np.zeros((len(peaks), bayes_n_nodes))

    num_samples = peaks['data'].shape[1]
    step_size = int(num_samples/bayes_n_nodes)
    steps = np.arange(0, num_samples+1, step_size)

    data = peaks['data'].copy()
    data[data < 0.0] = 0.0
    for i, p in enumerate(peaks):
        sample_number = np.arange(0, num_samples+1, 1)*p['dt']
        # create buffer to avoid np.append
        frac_of_cumsum = np.empty(num_samples+1)
        frac_of_cumsum[0] = 0.0
        frac_of_cumsum[1:] = np.cumsum(data[i, :]) / np.sum(data[i, :])
        # create buffer to avoid np.append
        cumsum_steps = np.empty(bayes_n_nodes+1)
        cumsum_steps[0:bayes_n_nodes] = np.interp(np.linspace(0., 1., bayes_n_nodes, endpoint=False), frac_of_cumsum, sample_number)
        cumsum_steps[-1:] = sample_number[-1]
        quantiles[i, :] = cumsum_steps[1:] - cumsum_steps[:-1]

    for j in range(bayes_n_nodes):
        waveforms[:, j] = np.sum(data[:, steps[j]:steps[j+1]], axis=1)
    waveforms = waveforms/(peaks['dt']*step_size)[:, np.newaxis]
    del data
    
    return waveforms, quantiles

def compute_inference(bins: int, bayes_n_nodes: int, cpt: np.ndarray, n_bayes_classes: int, class_prior: np.ndarray,
                      waveforms: np.ndarray, quantiles: np.ndarray):
    """
    Bin the waveforms and quantiles according to Bayes bins and compute inference
    :param bins: Bayes bins
    :param bayes_n_nodes: number of nodes or atributes
    :param cpt: conditioanl probability tables
    :param n_bayes_classes: number of classes
    :param class_prior: class_prior
    :param waveforms: waveforms
    :param quantiles: quantiles
    :return: ln probability per class S1/S2
    """
    # Bin the waveforms and quantiles.
    waveform_bin_edges = bins[0, :][bins[0, :] > -1]
    waveform_num_bin_edges = len(waveform_bin_edges)
    quantile_bin_edges = bins[1, :][bins[1, :] > -1]
    quantile_num_bin_edges = len(quantile_bin_edges)

    waveform_values = np.digitize(waveforms, bins=waveform_bin_edges)-1
    waveform_values[waveform_values < 0] = int(0)
    waveform_values[waveform_values > int(waveform_num_bin_edges - 2)] = int(waveform_num_bin_edges - 2)

    quantile_values = np.digitize(quantiles, bins=quantile_bin_edges)-1
    quantile_values[quantile_values < 0] = int(0)
    quantile_values[quantile_values > int(quantile_num_bin_edges - 2)] = int(quantile_num_bin_edges - 2)

    values_for_inference = np.append(waveform_values, quantile_values, axis=1)

    # Inference
    distributions = [[] for i in range(bayes_n_nodes*2)]
    for i in np.arange(0, bayes_n_nodes, 1):
        distributions[i] = np.asarray(cpt[i, :waveform_num_bin_edges-1, :])
    for i in np.arange(bayes_n_nodes, bayes_n_nodes * 2, 1):
        distributions[i] = np.asarray(cpt[i, :quantile_num_bin_edges-1, :])

    lnposterior = np.zeros((len(waveforms), bayes_n_nodes*2, n_bayes_classes))
    for i in range(bayes_n_nodes*2):
        lnposterior[:, i, :] = np.log(distributions[i][values_for_inference[:, i], :])

    lnposterior_sumsamples = np.sum(lnposterior, axis=1)
    lnposterior_sumsamples = np.sum([lnposterior_sumsamples, np.log(class_prior)[np.newaxis, ...]])
    lnposterior_normed = lnposterior_sumsamples - logsumexp(lnposterior_sumsamples, axis=1)[..., np.newaxis]

    return lnposterior_normed[:, 0], lnposterior_normed[:, 1]
