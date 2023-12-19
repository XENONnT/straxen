import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.spatial.distance import cdist
from straxen.plugins.peaklets.peaklet_classification import PeakletClassification
import numba

import strax
import straxen

export, __all__ = strax.exporter()


@export
class PeakletClassificationSOM(PeakletClassification):
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

    __version__ = "0.2.0"
    child_plugin = True

    som_files = straxen.URLConfig(
        default="resource://xedocs://som_classifiers?attr=value&version=v1&run_id=045000&fmt=npy"
    )

    use_som_as_default = straxen.URLConfig(
        default=True,
        help=(
            "Boolean to indicate whether to use SOM"
            " or the straxen classifcation method for"
            "peaklet classification"
        ),
    )

    def infer_dtype(self):
        dtype = strax.peak_interval_dtype + [
            ("type", np.int8, "Classification of the peak(let)"),
            ("som_sub_type", np.int32, "SOM subtype of the peak(let)"),
            ("straxen_type", np.int8, "Old straxen type of the peak(let)"),
            ("som_type", np.int8, "SOM type of the peak(let)"),
            ("loc_x_som", np.int16, "x location of the peak(let) in the SOM"),
            ("loc_y_som", np.int16, "y location of the peak(let) in the SOM"),
        ]
        return dtype

    def setup(self):
        self.som_weight_cube = self.som_files["weight_cube"]
        self.som_img = self.som_files["som_img"]
        self.som_norm_factors = self.som_files["norm_factors"]
        self.som_s1_array = self.som_files["s1_array"]
        self.som_s2_array = self.som_files["s2_array"]
        self.som_s3_array = self.som_files["s3_array"]
        self.som_s0_array = self.som_files["s0_array"]

    def compute(self, peaklets):
        # Current classification
        peaklets_classifcation = super().compute(peaklets)

        peaklet_with_som = np.zeros(len(peaklets_classifcation), dtype=self.dtype)
        strax.copy_to_buffer(peaklets_classifcation, peaklet_with_som, "_copy_peaklets_information")
        peaklet_with_som["straxen_type"] = peaklets_classifcation["type"]
        del peaklets_classifcation

        # SOM classification
        peaklets_w_type = peaklets.copy()
        peaklets_w_type["type"] = peaklet_with_som["type"]
        _is_s1_or_s2 = peaklets_w_type["type"] != 0

        peaklets_w_type = peaklets_w_type[_is_s1_or_s2]

        som_type, x_som, y_som = recall_populations(
            peaklets_w_type, self.som_weight_cube, self.som_img, self.som_norm_factors
        )
        peaklet_with_som["som_sub_type"][_is_s1_or_s2] = som_type
        peaklet_with_som["loc_x_som"][_is_s1_or_s2] = x_som
        peaklet_with_som["loc_y_som"][_is_s1_or_s2] = y_som

        strax_type = som_type_to_type(
            som_type, self.som_s1_array, self.som_s2_array, self.som_s3_array, self.som_s0_array
        )

        peaklet_with_som["som_type"][_is_s1_or_s2] = strax_type
        if self.use_som_as_default:
            peaklet_with_som["type"][_is_s1_or_s2] = strax_type
        else:
            peaklet_with_som["type"] = peaklet_with_som["straxen_type"]

        return peaklet_with_som


def recall_populations(dataset, weight_cube, som_cls_img, norm_factors):
    """Master function that should let the user provide a weightcube, a reference img as a np.array,
    a dataset and a set of normalization factors.

    In theory, if these 5 things are provided, this function should output
    the original data back with one added field with the name "SOM_type"
    weight_cube:      SOM weight cube (3D array)
    som_cls_img:      SOM reference image as a numpy array
    dataset:          Data to preform the recall on (Should be peaklet level data)
    normfactos:       A set of 11 numbers to normalize the data so we can preform a recall

    """

    xdim, ydim, zdim = weight_cube.shape
    img_xdim, img_ydim, img_zdim = som_cls_img.shape
    unique_colors = np.unique(np.reshape(som_cls_img, [xdim * ydim, 3]), axis=0)
    # Checks that the reference image matches the weight cube
    assert (
        xdim == img_xdim
    ), f"Dimensions mismatch between SOM weight cube ({xdim}) and reference image ({img_xdim})"
    assert (
        ydim == img_ydim
    ), f"Dimensions mismatch between SOM weight cube ({ydim}) and reference image ({img_ydim})"

    assert all(dataset["type"] != 0), "Dataset contains unclassified peaklets"
    # Get the deciles representation of data for recall
    decile_transform_check = data_to_log_decile_log_area_aft(dataset, norm_factors)
    # preform a recall of the dataset with the weight cube
    # assign each population color a number (can do from previous function)
    ref_map = generate_color_ref_map(som_cls_img, unique_colors, xdim, ydim)
    som_cls_array = np.empty(len(dataset["area"]))
    som_cls_array[:] = np.nan
    # Make new numpy structured array to save the SOM cls data
    data_with_SOM_cls = rfn.append_fields(dataset, "SOM_type", som_cls_array)
    # preforms the recall and assigns SOM_type label
    output_data, x_som, y_som = som_cls_recall(
        data_with_SOM_cls, decile_transform_check, weight_cube, ref_map
    )
    return output_data["SOM_type"], x_som, y_som


def generate_color_ref_map(color_image, unique_colors, xdim, ydim):
    ref_map = np.zeros((xdim, ydim))
    for color in np.arange(len(unique_colors)):
        mask = np.all(np.equal(color_image, unique_colors[color, :]), axis=2)
        indices = np.argwhere(mask)  # generates a 2d mask
        for loc in np.arange(len(indices)):
            ref_map[indices[loc][0], indices[loc][1]] = color
    return ref_map


def som_cls_recall(array_to_fill, data_in_som_fmt, weight_cube, reference_map):
    som_xdim, som_ydim, _ = weight_cube.shape
    # for data_point in data_in_SOM_fmt:
    distances = cdist(
        weight_cube.reshape(-1, weight_cube.shape[-1]), data_in_som_fmt, metric="euclidean"
    )
    w_neuron = np.argmin(distances, axis=0)
    x_idx, y_idx = np.unravel_index(w_neuron, (som_xdim, som_ydim))
    array_to_fill["SOM_type"] = reference_map[x_idx, y_idx]
    return array_to_fill, x_idx, y_idx


def som_type_to_type(som_type, s1_array, s2_array, s3_array, s0_array):
    """
    Converts the SOM type into either S1 or S2 type (1, 2)
    som_type:    array with integers corresponding to the different SOM types
    s1_array:    array containing the number corresponding to the SOM types which should
                 be converted to S1's
    """
    som_type_copy = som_type.copy()
    som_type_copy[np.isin(som_type_copy, s1_array)] = 1234
    som_type_copy[np.isin(som_type_copy, s2_array)] = 5678
    som_type_copy[np.isin(som_type_copy, s3_array)] = -5
    som_type_copy[np.isin(som_type_copy, s0_array)] = -250
    som_type_copy[som_type_copy == 1234] = 1
    som_type_copy[som_type_copy == 5678] = 2
    som_type_copy[som_type_copy == -5] = 3
    som_type_copy[som_type_copy == -250] = 0

    return som_type_copy


def data_to_log_decile_log_area_aft(peaklet_data, normalization_factor):
    """Converts peaklet data into the current best inputs for the SOM, log10(deciles) + log10(area)
    + AFT Since we are dealing with logs, anything less than 1 will be set to 1."""
    # turn deciles into approriate 'normalized' format
    # (maybe also consider L1 normalization of these inputs)
    decile_data = compute_quantiles(peaklet_data, 10)
    data = peaklet_data.copy()
    decile_data[decile_data < 1] = 1

    decile_log = np.log10(decile_data)
    decile_log_over_max = np.divide(decile_log, normalization_factor[:10])
    # Now lets deal with area
    data["area"] = data["area"] + normalization_factor[11] + 1
    peaklet_log_area = np.log10(data["area"])
    peaklet_aft = (
        np.sum(data["area_per_channel"][:, : straxen.n_top_pmts], axis=1) / peaklet_data["area"]
    )
    peaklet_aft = np.where(peaklet_aft > 0, peaklet_aft, 0)
    peaklet_aft = np.where(peaklet_aft < 1, peaklet_aft, 1)
    deciles_area_aft = np.concatenate(
        (
            decile_log_over_max,
            np.reshape(peaklet_log_area, (len(peaklet_log_area), 1)) / normalization_factor[10],
            np.reshape(peaklet_aft, (len(peaklet_log_area), 1)),
        ),
        axis=1,
    )
    return deciles_area_aft


def compute_quantiles(peaks: np.ndarray, n_samples: int):
    """Compute waveforms and quantiles for a given number of nodes(attributes) :param peaks:

    :param n_samples: number of nodes or attributes
    :return: quantiles

    """

    data = peaks["data"].copy()
    data[data < 0.0] = 0.0
    dt = peaks["dt"]
    q = compute_wf_attributes(data, dt, n_samples)
    return q


@export
@numba.jit(nopython=True, cache=True)
def compute_wf_attributes(data, sample_length, n_samples: int):
    """
    Compute waveform attribures
    Quantiles: represent the amount of time elapsed for
    a given fraction of the total waveform area to be observed in n_samples
    i.e. n_samples = 10, then quantiles are equivalent deciles
    Waveforms: downsampled waveform to n_samples
    :param data: waveform e.g. peaks or peaklets
    :param n_samples: compute quantiles for a given number of samples
    :return: waveforms and quantiles of size n_samples
    """

    assert data.shape[0] == len(sample_length), "ararys must have same size"

    num_samples = data.shape[1]

    quantiles = np.zeros((len(data), n_samples), dtype=np.float64)

    # Cannot compute with with more samples than actual waveform sample
    assert num_samples > n_samples, "cannot compute with more samples than the actual waveform"
    assert num_samples % n_samples == 0, "number of samples must be a multiple of n_samples"

    # Compute quantiles
    inter_points = np.linspace(0.0, 1.0 - (1.0 / n_samples), n_samples)
    cumsum_steps = np.zeros(n_samples + 1, dtype=np.float64)
    frac_of_cumsum = np.zeros(num_samples + 1)
    sample_number_div_dt = np.arange(0, num_samples + 1, 1)
    for i, (samples, dt) in enumerate(zip(data, sample_length)):
        if np.sum(samples) == 0:
            continue
        # reset buffers
        frac_of_cumsum[:] = 0
        cumsum_steps[:] = 0
        frac_of_cumsum[1:] = np.cumsum(samples)
        frac_of_cumsum[1:] = frac_of_cumsum[1:] / frac_of_cumsum[-1]
        cumsum_steps[:-1] = np.interp(inter_points, frac_of_cumsum, sample_number_div_dt * dt)
        cumsum_steps[-1] = sample_number_div_dt[-1] * dt
        quantiles[i] = cumsum_steps[1:] - cumsum_steps[:-1]

    return quantiles


@export
class PeakletSOMClass(PeakletClassificationSOM):
    """Plugin which allows in addition to the straxen classification the SOM classification."""

    child_plugin = True
    __version__ = "0.0.1"

    provides = "peaklet_classification_som"

    def compute(self, peaklets):
        peaklet_classifcation_som = super().compute(peaklets)
        return peaklet_classifcation_som
