from .url_config import URLConfig

import os
import json
import pytz
import typing
import strax
import fsspec
import straxen
import tarfile
import tempfile
from typing import Container, Iterable, Optional

import numpy as np

from immutabledict import immutabledict

from utilix import xent_collection
import utilix
from scipy.interpolate import interp1d


def get_item_or_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@URLConfig.register("cmt")
def get_correction(
    name: str, run_id: Optional[str] = None, version: str = "ONLINE", detector: str = "nt", **kwargs
):
    """Get value for name from CMT."""

    if run_id is None:
        raise ValueError("Attempting to fetch a correction without a run_id.")

    return straxen.get_correction_from_cmt(run_id, (name, version, detector == "nt"))


@URLConfig.register("resource")
def get_resource(name: str, fmt: str = "text", **kwargs):
    """Fetch a straxen resource Allow a direct download using <fmt='abs_path'> otherwise kwargs are
    passed directly to straxen.get_resource."""
    if fmt == "abs_path":
        downloader = utilix.mongo_storage.MongoDownloader()
        return downloader.download_single(name)
    return straxen.get_resource(name, fmt=fmt)


@URLConfig.register("fsspec")
def read_file(path: str, **kwargs):
    """Support fetching files from arbitrary filesystems."""
    with fsspec.open(path, **kwargs) as f:
        content = f.read()
    return content


@URLConfig.register("json")
def read_json(content: str, **kwargs):
    """Load json string as a python object."""
    return json.loads(content)


@URLConfig.register("take")
def get_key(container: Container, take=None, **kwargs):
    """Return a single element of a container."""
    if take is None:
        return container
    if not isinstance(take, list):
        take = [take]

    # support for multiple keys for
    # nested objects
    for t in take:
        container = container[t]  # type: ignore

    return container


@URLConfig.register("format")
def format_arg(arg: str, **kwargs):
    """Apply pythons builtin format function to a string."""
    return arg.format(**kwargs)


@URLConfig.register("itp_map")
def load_map(some_map, method="WeightedNearestNeighbors", scale_coordinates=None, **kwargs):
    """Make an InterpolatingMap."""
    itp_map = straxen.InterpolatingMap(some_map, method=method, **kwargs)
    if scale_coordinates is not None:
        itp_map.scale_coordinates(scale_coordinates)
    return itp_map


@URLConfig.register("bodega")
def load_value(name: str, bodega_version=None):
    """Load a number from BODEGA file."""
    if bodega_version is None:
        raise ValueError("Provide version see e.g. tests/test_url_config.py")
    nt_numbers = straxen.get_resource("XENONnT_numbers.json", fmt="json")
    return nt_numbers[name][bodega_version]["value"]


@URLConfig.register("tf")
def open_neural_net(model_path: str, custom_objects=None, **kwargs):
    """Open a tensorflow file and return a keras model."""
    # Nested import to reduce loading time of import straxen and it not
    # base requirement
    import tensorflow as tf

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No file at {model_path}")
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar = tarfile.open(model_path, mode="r:gz")
        tar.extractall(path=tmpdirname)
        return tf.keras.models.load_model(tmpdirname, custom_objects=custom_objects)


@URLConfig.register("itp_dict")
def get_itp_dict(
    loaded_json, run_id=None, time_key="time", itp_keys="correction", **kwargs
) -> typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]:
    """Interpolate a dictionary at the start time that is queried from a run-id.

    :param loaded_json: a dictionary with a time-series
    :param run_id: run_id
    :param time_key: key that gives the timestamps
    :param itp_keys: which keys from the dict to read. Should be comma (',') separated!
    :return: Interpolated values of dict at the start time, either returned as an np.ndarray (single
        value) or as a dict (multiple itp_dict_keys)

    """
    keys = strax.to_str_tuple(itp_keys.split(","))
    for key in list(keys) + [time_key]:
        if key not in loaded_json:
            raise KeyError(
                f"The json does contain the key '{key}'. Try one of: {loaded_json.keys()}"
            )

    times = loaded_json[time_key]

    # get start time of this run. Need to make tz-aware
    start = xent_collection().find_one({"number": int(run_id)}, {"start": 1})["start"]
    start = pytz.utc.localize(start).timestamp() * 1e9

    try:
        if len(strax.to_str_tuple(keys)) > 1:
            return {
                key: interp1d(times, loaded_json[key], bounds_error=True)(start) for key in keys
            }

        else:
            interp = interp1d(times, loaded_json[keys[0]], bounds_error=True)
            return interp(start)
    except ValueError as e:
        raise ValueError(f"Correction is not defined for run {run_id}") from e


@URLConfig.register("rekey_dict")
def rekey_dict(d, replace_keys="", with_keys=""):
    """Replace the keys of a dictionary.

    :param d: dictionary that will have its keys renamed
    :param replace_keys: comma-separated string of keys that will be replaced
    :param with_keys: comma-separated string of keys that will replace the replace_keys
    :return: dictionary with renamed keys

    """
    new_dict = d.copy()
    replace_keys = strax.to_str_tuple(replace_keys.split(","))
    with_keys = strax.to_str_tuple(with_keys.split(","))
    if len(replace_keys) != len(with_keys):
        raise RuntimeError("replace_keys and with_keys must have the same length")
    for old_key, new_key in zip(replace_keys, with_keys):
        new_dict[new_key] = new_dict.pop(old_key)
    return new_dict


@URLConfig.register("objects-to-dict")
def objects_to_dict(objects: list, key_attr=None, value_attr="value", immutable=False):
    """Converts a list of objects/dicts to a single dictionary by taking the key and value from each
    of the objects/dicts. If key_attr is not provided, the list index is used as the key.

    :param objects: list of objects/dicts that will be converted to a dictionary
    :param key_attr: key/attribute of the objects that will be used as key in the dictionary
    :param value_attr: key/attribute of the objects that will be used as value in the dictionary

    """
    if not isinstance(objects, Iterable):
        raise TypeError(
            f"The objects-to-dict protocol expects an iterable "
            f"of objects but received {type(objects)} instead."
        )
    result = {}
    for i, obj in enumerate(objects):
        key = i if key_attr is None else get_item_or_attr(obj, key_attr)
        result[key] = get_item_or_attr(obj, value_attr)

    if immutable:
        result = immutabledict(result)

    return result


@URLConfig.register("list-to-array")
def objects_to_array(objects: list):
    """Converts a list of objects/dicts to a numpy array.

    :param objects: Any list of objects

    """

    if not isinstance(objects, Iterable):
        raise TypeError(
            f"The list-to-array protocol expects an "
            f"iterable but recieved a {type(objects)} instead"
        )

    return np.array(objects)


@URLConfig.register("run_doc")
def read_rundoc(path, run_id=None, default=None):
    """Read a path from the json rundoc metadata.

    :param path: keys of json rundoc metada.
        e.g. `comments` for reading comments for a specific run

    """
    if run_id is None:
        raise ValueError("rundoc protocol: missing run_id.")
    runs = xent_collection()
    rundoc = runs.find_one({"number": int(run_id)}, {"_id": 0, path: 1})
    if rundoc is None:
        raise ValueError(f"No rundoc found for run {run_id}")

    for part in path.split("."):
        if isinstance(rundoc, list) and part.isdigit() and len(rundoc) > int(part):
            rundoc = rundoc[int(part)]
        elif isinstance(rundoc, dict) and part in rundoc:
            rundoc = rundoc[part]
        elif default is not None:
            return default
        else:
            raise ValueError(f"No path {path} found in rundoc for run {run_id}")
    return rundoc


@URLConfig.register("pad-array")
def get_paded_array(arr: np.ndarray, pad_value=0, pad_left=0, pad_right=0):
    """Pad the array with pad_value on the left and right side."""
    return np.pad(arr, (pad_left, pad_right), constant_values=pad_value)


@URLConfig.register("jax")
def open_jax_model(model_path: str, **kwargs):
    """Open and deserialize a JAX model from a tar file.

    This function is registered with straxen's URLConfig under the 'jax' key.
    It opens a tar file containing serialized JAX models, extracts the requested
    model based on the provided parameters, and returns the deserialized model.

    Args:
        model_path (str): Path to the tar file containing the JAX models.
        **kwargs: Additional keyword arguments.
            Required:
                n_poly (int): Number of polynomials used in the model.
                sig (float): Sigma value used in the model.

    Returns:
        callable: The deserialized JAX model as a callable function.

    Raises:
        FileNotFoundError: If the specified model_path does not exist.
        ValueError: If the requested model is not found in the tar file.

    Note:
        This function uses a nested import of 'jax.export' to reduce the loading
        time of importing straxen, as JAX is not a base requirement.

    """
    # Nested import to reduce loading time of import straxen as it's not
    # a base requirement
    from jax import export

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No file at {model_path}")

    # Extract required parameters from kwargs
    n_poly = kwargs["n_poly"]
    sig = kwargs["sig"]

    # Open the tar file and extract the requested model
    with tarfile.open(model_path, "r") as f:
        names = f.getnames()

        # Construct the filename based on n_poly and sig
        filename = f"{int(n_poly):02d}_{float(sig) * 1000:3.0f}"

        # Check if the requested model exists in the tar file
        if filename not in names:
            raise ValueError("Requested model not in tarfile!")

        # Extract and read the serialized JAX object
        file_obj = f.extractfile(filename)
        assert file_obj is not None
        serialized_jax_object = file_obj.read()
    # Deserialize the JAX object and return its callable function
    return export.deserialize(serialized_jax_object).call

@URLConfig.register("keras3")
def open_neural_net(model_path: str, custom_objects=None, **kwargs):
    """Load a Keras model from a Keras file."""
    # Nested import to reduce loading time of import straxen and it not
    # base requirement
    import tensorflow as tf

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No file at {model_path}")

    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)