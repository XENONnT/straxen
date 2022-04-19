import numpy as np
import numba
import strax
import straxen


class BayesPeakClassification(strax.Plugin):
    """
    Bayes Peak classification
    Returns the ln probability of a each event belonging to the S1 and S2 class.
    Uses conditional probabilities and data parameterization learned from wfsim data.
    More info can be found here xenon:xenonnt:ahiguera:bayespeakclassification

    :param peaks: peaks
    :param waveforms: peaks waveforms in PE/ns
    :param quantiles: quantiles in ns, calculate from a cumulative sum over the waveform,
                      from zero to the total area with normalized cumulative sum to determine the time
    :returns: the ln probability of a each peak belonging to S1 and S2 class
    """

    provides = 'peak_classification_bayes'
    depends_on = ('peaks',)
    __version__ = '0.0.3'
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
        result = np.zeros(len(peaks), dtype=self.dtype)

        waveforms, quantiles = compute_wf_and_quantiles(peaks, self.bayes_n_nodes)

        ln_prob_s1, ln_prob_s2 = compute_inference(self.bins, self.bayes_n_nodes, self.cpt,
                                                   self.n_bayes_classes, self.class_prior,
                                                   waveforms, quantiles)
        result['time'] = peaks['time']
        result['endtime'] = peaks['time'] + peaks['dt'] * peaks['length']
        result['ln_prob_s1'] = ln_prob_s1
        result['ln_prob_s2'] = ln_prob_s2
        return result
    
    
def compute_wf_and_quantiles(peaks: np.ndarray, bayes_n_nodes: int):
    """
    Compute waveforms and quantiles for a given number of nodes(atributes)
    :param peaks:
    :param bayes_n_nodes: number of nodes or atributes
    :return: waveforms and quantiles
    """
    data = peaks['data'].copy()
    data[data < 0.0] = 0.0
    dt = peaks['dt']
    return _compute_wf_and_quantiles(data, dt, bayes_n_nodes)
    

@numba.njit(cache=True)
def _compute_wf_and_quantiles(data, sample_length, bayes_n_nodes: int):
    waveforms = np.zeros((len(data), bayes_n_nodes))
    quantiles = np.zeros((len(data), bayes_n_nodes))

    num_samples = data.shape[1]
    step_size = int(num_samples/bayes_n_nodes)
    inter_points = np.linspace(0., 1.-(1./bayes_n_nodes), bayes_n_nodes)
    cumsum_steps = np.zeros(bayes_n_nodes + 1, dtype=np.float64)
    frac_of_cumsum = np.zeros(num_samples + 1)
    sample_number_div_dt = np.arange(0, num_samples+1, 1)
    for i, (waveform, dt) in enumerate(zip(data, sample_length)):
        # reset buffers
        frac_of_cumsum[:] = 0
        cumsum_steps[:] = 0
        
        frac_of_cumsum[1:] = np.cumsum(waveform)
        frac_of_cumsum[1:] = frac_of_cumsum[1:]/frac_of_cumsum[-1]
        
        cumsum_steps[:-1] = np.interp(inter_points, frac_of_cumsum, sample_number_div_dt * dt)
        cumsum_steps[-1] = sample_number_div_dt[-1] * dt
        quantiles[i] = cumsum_steps[1:] - cumsum_steps[:-1]

        for j in range(bayes_n_nodes):
            for k in range(j, j+1):
                waveforms[i][j] += waveform[k]
        waveforms[i] /= (step_size*dt)
    
    return waveforms, quantiles


@numba.njit
def _get_log_posterior(nodes, n_bins, n_classes, cpt, wf_len, wf_values):
    ln_posterior = np.zeros((wf_len, nodes, n_classes))
    for i in range(nodes):
        distribution = cpt[i, :n_bins, :]
        ln_posterior[:, i, :] = np.log(distribution[wf_values[:, i], :])
    return ln_posterior


@numba.njit
def _logsumexp_axis1(arr, axis=1):
    """~20x faster than scipy.special.logsumexp(*, axis=1)"""
    if axis == 1:
        res = np.zeros(len(arr), dtype=np.float64)
        for i, a in enumerate(arr):
            res[i] = np.log(sum(np.e**a))
        return res
    raise ValueError


@numba.njit
def _set_2d_to_zero_or_max_val(values, max_val):
    for k, w in enumerate(values):
        for kk, ww in enumerate(w):
            if ww < 0:
                values[k][kk] = 0
            if ww > max_val:                
                values[k][kk] = max_val


@numba.njit(cache=True)
def compute_inference(bins: int, 
                      bayes_n_nodes: int,
                      cpt: np.ndarray,
                      n_bayes_classes: int,
                      class_prior: np.ndarray,
                      waveforms: np.ndarray, 
                      quantiles: np.ndarray):
    """
    Bin the waveforms and quantiles according to Bayes bins and compute inference
    :param bins: Bayes bins
    :param bayes_n_nodes: number of nodes or atributes
    :param cpt: conditional probability tables
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
    _set_2d_to_zero_or_max_val(waveform_values, np.int64(waveform_num_bin_edges - 2))

    quantile_values = np.digitize(quantiles, bins=quantile_bin_edges)-1
    _set_2d_to_zero_or_max_val(quantile_values, np.int64(quantile_num_bin_edges - 2))

    wf_posterior = _get_log_posterior(
        nodes=bayes_n_nodes, 
        n_bins=waveform_num_bin_edges-1,
        n_classes=n_bayes_classes,
        cpt=cpt[:bayes_n_nodes],
        wf_len=len(waveforms),
        wf_values=waveform_values
    )
    quantile_posterior = _get_log_posterior(
        nodes=bayes_n_nodes, 
        n_bins=quantile_num_bin_edges-1,
        n_classes=n_bayes_classes,
        cpt=cpt[bayes_n_nodes:],
        wf_len=len(waveforms),
        wf_values=quantile_values
    )

    lnposterior = np.zeros((len(waveforms), bayes_n_nodes*2, n_bayes_classes))
    lnposterior[:, :bayes_n_nodes] = wf_posterior
    lnposterior[:, bayes_n_nodes:] = quantile_posterior
    lnposterior_sumsamples = np.sum(lnposterior, axis=1)
    lnposterior_sumsamples = lnposterior_sumsamples + np.log(class_prior)

    normalization = _logsumexp_axis1(lnposterior_sumsamples, axis=1)
    ln_prob_s1 = lnposterior_sumsamples[:, 0]-normalization
    ln_prob_s2 = lnposterior_sumsamples[:, 1]-normalization
    return ln_prob_s1, ln_prob_s2
