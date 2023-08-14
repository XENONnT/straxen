import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.spatial.distance import cdist

import strax
import straxen

export, __all__ = strax.exporter()


@export
class PeakletClassificationSOM(strax.Plugin):
    """
    Self-Organizing Maps (SOM)
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:lsanchez:unsupervised_neural_network_som_methods
    For peaklet classification. We this pluggin will provide 2 data types, the 'type' we are
    already familiar with, classifying peaklets as s1, s2 (using the new classification) or
    unknown (from the previous classification). As well as a new data type, SOM type, which
    will be assigned numbers based on the cluster in the SOM in which they are found. For
    each version I will make some documentation in the corrections repository explaining
    what I believe each cluster represents.

    This correction/plugin is currently on the testing phase, feel free to use it if you are
    curious or just want to test it or try it out but note this is note ready to be used in
    analysis.
    """
    __version__ = '0.0.1'

    provides = 'peaklet_classification_som'
    depends_on = ('peaklets', 'peaklet_classification')
    parallel = True
    dtype = (strax.peak_interval_dtype +
             [('type', np.int8, 'Classification of the peak(let)')]) 

    som_files = straxen.URLConfig(default='resource:///stor2/data/LS_data/SOM_data/som_data_v0.npz?fmt=npy')

    def setup(self):
        self.som_weight_cube = self.som_files['weight_cube']
        self.som_img = self.som_files['som_img']
        self.som_norm_factors = self.som_files['norm_factors']
        self.som_s1_array = self.som_files['s1_array']
        self.som_s2_array = self.som_files['s2_array']

    def compute(self, peaklets):
        peaklets_w_type = peaklets.copy()
        mask_non_zero = peaklets_w_type['type'] != 0
        peaklets_w_type = peaklets_w_type[mask_non_zero]
        result = np.zeros(len(peaklets), dtype=self.dtype)
        som_type = recall_populations(peaklets_w_type, self.som_weight_cube,
                                      self.som_img,
                                      self.som_norm_factors)
        strax_type = som_type_to_type(som_type,
                                      self.som_s1_array,
                                      self.som_s2_array)
        result['time'] = peaklets['time']
        result['length'] = peaklets['length']
        result['dt'] = peaklets['dt']
        result['type'][mask_non_zero] = strax_type
        return result


def recall_populations(dataset, weight_cube, SOM_cls_img, norm_factors):
    """
    Master function that should let the user provide a weightcube,
    a reference img as a np.array, a dataset and a set of normalization factors.
    In theory, if these 5 things are provided, this function should output
    the original data back with one added field with the name "SOM_type"
    weight_cube:      SOM weight cube (3D array)
    SOM_cls_img:      SOM reference image as a numpy array
    dataset:          Data to preform the recall on (Should be peaklet level data)
    normfactos:       A set of 11 numbers to normalize the data so we can preform a recall
    """
    [SOM_xdim, SOM_ydim, SOM_zdim] = weight_cube.shape
    [IMG_xdim, IMG_ydim, IMG_zdim] = SOM_cls_img.shape
    unique_colors = np.unique(np.reshape(SOM_cls_img, [SOM_xdim * SOM_ydim, 3]), axis=0)
    # Checks that the reference image matches the weight cube
    assert SOM_xdim == IMG_xdim, f'Dimensions mismatch between SOM weight cube ({SOM_xdim}) and reference image ({IMG_xdim})'
    assert SOM_ydim == IMG_ydim, f'Dimensions mismatch between SOM weight cube ({SOM_ydim}) and reference image ({IMG_ydim})'

    assert all(dataset['type'] != 0), 'Dataset contains unclassified peaklets'
    # Get the deciles representation of data for recall
    decile_transform_check = data_to_log_decile_log_area_aft(dataset, norm_factors)
    # preform a recall of the dataset with the weight cube
    # assign each population color a number (can do from previous function)
    ref_map = generate_color_ref_map(SOM_cls_img, unique_colors, SOM_xdim, SOM_ydim)
    SOM_cls_array = np.empty(len(dataset['area']))
    SOM_cls_array[:] = np.nan
    # Make new numpy structured array to save the SOM cls data
    data_with_SOM_cls = rfn.append_fields(dataset, 'SOM_type', SOM_cls_array)
    # preforms the recall and assigns SOM_type label
    output_data = SOM_cls_recall(data_with_SOM_cls, decile_transform_check, weight_cube, ref_map)
    return output_data['SOM_type']


def generate_color_ref_map(color_image, unique_colors, xdim, ydim):
    ref_map = np.zeros((xdim, ydim))
    for color in np.arange(len(unique_colors)):
        mask = np.all(np.equal(color_image, unique_colors[color, :]), axis=2)
        indices = np.argwhere(mask)  # generates a 2d mask
        for loc in np.arange(len(indices)):
            ref_map[indices[loc][0], indices[loc][1]] = color
    return ref_map


def SOM_cls_recall(array_to_fill, data_in_SOM_fmt, weight_cube, reference_map):
    [SOM_xdim, SOM_ydim, _] = weight_cube.shape
    # for data_point in data_in_SOM_fmt:
    distances = cdist(weight_cube.reshape(-1, weight_cube.shape[-1]), data_in_SOM_fmt, metric='euclidean')
    w_neuron = np.argmin(distances, axis=0)
    x_idx, y_idx = np.unravel_index(w_neuron, (SOM_xdim, SOM_ydim))
    array_to_fill['SOM_type'] = reference_map[x_idx, y_idx]
    return array_to_fill


def som_type_to_type(som_type, s1_array, s2_array):
    """
    Converts the SOM type into either S1 or S2 type (1, 2)
    som_type:    array with integers corresponding to the different SOM types
    s1_array:    array containing the number corresponding to the SOM types which should
                 be converted to S1's
    """
    som_type_copy = som_type.copy()
    assert len(np.unique(som_type)) == (len(s1_array) + len(s2_array)), f'Error, the number of SOM types provided ({len(np.unique(som_type_copy))}) does not match' f'the arrays to convert to ({len(s1_array) + len(s2_array)})'
    som_type_copy[np.isin(som_type_copy, s1_array)] = 1234
    som_type_copy[np.isin(som_type_copy, s2_array)] = 5678
    som_type_copy[som_type_copy == 1234] = 1
    som_type_copy[som_type_copy == 5678] = 2
    assert np.all(np.unique(som_type_copy) == np.array([1, 2])), f'Error, values other than s1 and s2 found in the array'
    return som_type_copy


# Need function to convert things to S1s and S2s
def data_to_log_decile_log_area_aft(peaklet_data, normalization_factor):
    """
    Converts peaklet data into the current best inputs for the SOM,
    log10(deciles) + log10(area) + AFT
    Since we are dealing with logs, anything less than 1 will be set to 1
    """
    # turn deciles into approriate 'normalized' format (maybe also consider L1 normalization of these inputs)
    decile_data = compute_quantiles(peaklet_data, 10)
    data = peaklet_data.copy()
    decile_data[decile_data < 1] = 1
    # decile_L1 = np.log10(decile_data)
    decile_log = np.log10(decile_data)
    decile_log_over_max = np.divide(decile_log, normalization_factor[:10])
    # Now lets deal with area
    data['area'] = data['area'] + normalization_factor[11] + 1
    peaklet_log_area = np.log10(data['area'])
    peaklet_aft = np.sum(data['area_per_channel'][:, :straxen.n_top_pmts], axis=1) / normalization_factor[10]
    peaklet_aft = np.where(peaklet_aft > 0, peaklet_aft, 0)
    peaklet_aft = np.where(peaklet_aft < 1, peaklet_aft, 1)
    deciles_area_aft = np.concatenate((decile_log_over_max,
                                       np.reshape(peaklet_log_area, (len(peaklet_log_area), 1)) / normalization_factor[
                                           10],
                                       np.reshape(peaklet_aft, (len(peaklet_log_area), 1))), axis=1)
    return deciles_area_aft


def compute_quantiles(peaks: np.ndarray, n_samples: int):
    """
    Compute waveforms and quantiles for a given number of nodes(attributes)
    :param peaks:
    :param n_samples: number of nodes or attributes
    :return:quantiles
    """
    data = peaks['data'].copy()
    data[data < 0.0] = 0.0
    dt = peaks['dt']
    q, wf = strax.compute_wf_attributes(data, dt, n_samples, False)
    return q
