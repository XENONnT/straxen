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

    :param peaklets: peaklets
    :param waveforms: peaklets waveforms in PE/ns
    :param quantiles: quantiles in ns
    :returns: the ln probability of a each peaklet belonging to S1 and S2 class
    """
    provides = 'peaklet_classification_bayes'
    depends_on = ('peaklets',)
    __version__ = '0.0.1'
    dtype = (strax.peak_interval_dtype
             + [('s1_ln_prob', np.float32, 'S1 ln probability')]
             + [('s2_ln_prob', np.float32, 'S2 ln probability')]
            )

    # Descriptor configs
    bayes_config_file = straxen.URLConfig(
        default='resource://'
        'conditional_probabilities_and_bins_v1_w_global_v6.npz?fmt=npy',
        help='Bayes configuration file, conditional probabilities tables and Bayes discrete bins'
    )
    num_nodes = straxen.URLConfig(
        default=50,
        help='Number of attributes(features) per waveform and quantile'
    )
    classes = straxen.URLConfig(
        default=2,
        help='Number of label classes S1(1)/S2(2)'
    )

    def setup(self):

        self.class_prior = np.ones(self.classes)/self.classes
        self.bins = self.bayes_config_file['bins']
        self.cpt = self.bayes_config_file['cprob']

    def compute(self, peaklets):

        waveforms, quantiles = compute_wf_and_quantiles(peaklets, self.num_nodes)

        s1_ln_prob, s2_ln_prob = compute_inference(self.bins, self.num_nodes, self.cpt,
                                                   self.classes, self.class_prior,
                                                   waveforms, quantiles)

        return dict(time=peaklets['time'],
                    dt=peaklets['dt'],
                    channel=-1,
                    length=peaklets['length'],
                    s1_ln_prob=s1_ln_prob,
                    s2_ln_prob=s2_ln_prob
                    )


def compute_wf_and_quantiles(peaklets: np.ndarray, num_nodes: int):
    """
    Compute waveforms and quantiles for a given number of nodes(atributes)
    :param peaklets:
    :param num_nodes: number of nodes or atributes
    :return: waveforms and quantiles
    """
    waveforms = np.zeros((len(peaklets), num_nodes))
    quantiles = np.zeros((len(peaklets), num_nodes))

    num_samples = peaklets['data'].shape[1]
    step_size = int(num_samples/num_nodes)
    steps = np.arange(0, num_samples+1, step_size)

    data = peaklets['data'].copy()
    dts = peaklets['dt'].copy()
    data[data < 0.0] = 0.0
    for i, p in enumerate(peaklets):
        sample_number = np.arange(0, num_samples+1, 1)*p['dt']
        frac_of_cumsum = np.append([0.0], np.cumsum(data[i, :]) / np.sum(data[i, :]))
        cumsum_steps = np.interp(np.linspace(0., 1., num_nodes, endpoint=False), frac_of_cumsum, sample_number)
        cumsum_steps = np.append(cumsum_steps, sample_number[-1])
        quantiles[i, :] = cumsum_steps[1:] - cumsum_steps[:-1]

    for j in range(num_nodes):
        waveforms[:, j] = np.sum(data[:, steps[j]:steps[j+1]], axis=1)
    waveforms = waveforms/(dts*step_size)[:, np.newaxis]

    del data
    return waveforms, quantiles


def compute_inference(bins: int, num_nodes: int, cpt: np.ndarray, classes: int, class_prior: np.ndarray,
                      waveforms: np.ndarray, quantiles: np.ndarray):
    """
    Bin the waveforms and quantiles according to Bayes bins and compute inference
    :param bins: Bayes bins
    :param num_nodes: number of nodes or atributes
    :param cpt: conditioanl probability tables
    :param classes: number of classes
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
    distributions = [[] for i in range(num_nodes*2)]
    for i in np.arange(0, num_nodes, 1):
        distributions[i] = np.asarray(cpt[i, :waveform_num_bin_edges-1, :])
    for i in np.arange(num_nodes, num_nodes * 2, 1):
        distributions[i] = np.asarray(cpt[i, :quantile_num_bin_edges-1, :])

    lnposterior = np.zeros((len(waveforms), num_nodes*2, classes))
    for i in range(num_nodes*2):
        lnposterior[:, i, :] = np.log(distributions[i][values_for_inference[:, i], :])

    lnposterior_sumsamples = np.sum(lnposterior, axis=1)
    lnposterior_sumsamples = np.sum([lnposterior_sumsamples, np.log(class_prior)[np.newaxis, ...]])
    lnposterior_normed = lnposterior_sumsamples - logsumexp(lnposterior_sumsamples, axis=1)[..., np.newaxis]

    return lnposterior_normed[:, 0], lnposterior_normed[:, 1]
