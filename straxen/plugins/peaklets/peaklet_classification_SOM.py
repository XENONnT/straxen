import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.spatial.distance import cdist

import strax
import straxen


export, __all__ = strax.exporter()


@export
class PeakletClassificationSOM(strax.Plugin):
    """
    We use Self-Organizing Maps (SOM) https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:lsanchez:unsupervised_neural_network_som_methods
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
    dtype = (strax.peak_interval_dtype
             + [('type', np.int8, 'Classification of the peak(let)')]
             + [('som_type', np.int8, 'Classification of peaklets by SOM clusters')])

    SOM_files = straxen.URLConfig(
        default='resource://xedocs://'
                'som_nn'
                '?version=ONLINE&run_id=plugin.run_id&fmt=npy',
        help='SOM weightcube, normalization files, SOM class map array and convertion from SOM-type to type'
    )

    def compute(self, peaklets, peaklet_classification):

        peaklets_w_type = peaklets.copy()

        peaklets_w_type['type'] = peaklet_classification['type']

        result = np.zeros(len(peaklets_w_type), dtype=self.dtype)

        som_type = recall_populations(peaklets_w_type, self.SOM_files['weight_cube'],
                           self.SOM_files['som_img'], self.SOM_files['norm_factors'])

        strax_type = som_type_to_type(som_type,
                                self.SOM_files['s1_type'],
                                self.SOM_files['s2_type'])

        result['time'] = peaks['time']
        result['endtime'] = strax.endtime(peaks)
        result['som_type'] = som_type
        result['type'] = strax_type
        return result
    # Work on these functions to make them fit with a class
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

    assert len(np.unique(som_type)) == (len(s1_array) + len(s2_array)),
    f'Error, the number of SOM types provided ({len(np.unique(som_type_copy))}) does not match'
    f'the arrays to convert to ({len(s1_array) + len(s2_array)})'

    som_type_copy[np.isin(som_type_copy, s1_array)] = 1234
    som_type_copy[np.isin(som_type_copy, s2_array)] = 5678

    som_type_copy[som_type_copy == 1234] = 1
    som_type_copy[som_type_copy == 5678] = 2

    assert np.all(np.unique(som_type_copy) == np.array([1,2])), f'Error, values other than s1 and s2 found in the array'

    return som_type_copy
# Need function to convert things to S1s and S2s