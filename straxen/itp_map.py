import os
import re
import gzip
import json
import time
import pickle
import logging
import warnings
from typing import List, Literal, Callable, Union, Optional
from textwrap import dedent

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

import straxen
import strax

export, __all__ = strax.exporter()


@export
class InterpolateAndExtrapolate:
    """Linearly interpolate- and extrapolate using inverse-distance weighted averaging between
    nearby points."""

    def __init__(self, points, values, neighbours_to_use=None, array_valued=False):
        """
        :param points: array (n_points, n_dims) of coordinates
        :param values: array (n_points) of values
        :param neighbours_to_use: Number of neighbouring points to use for
        averaging. Default is 2 * dimensions of points.
        """
        self.kdtree = cKDTree(points)
        self.values = values
        if neighbours_to_use is None:
            neighbours_to_use = points.shape[1] * 2
        self.neighbours_to_use = neighbours_to_use
        self.array_valued = array_valued
        if array_valued:
            self.n_dim = self.values.shape[-1]

    def __call__(self, points):
        points = np.asarray(points)

        # kdtree doesn't grok NaNs
        # Start with all Nans, then overwrite for the finite points
        result = np.empty(len(points))
        # fill method slightly faster than multiplication of np.ones with nan
        result.fill(float("nan"))

        if self.array_valued:
            result = np.repeat(result.reshape(-1, 1), self.n_dim, axis=1)
        valid = np.all(np.isfinite(points), axis=-1)

        # Get distances to neighbours_to_use nearest neighbours
        distances, indices = self.kdtree.query(points[valid], self.neighbours_to_use)

        # Get values and weights for inverse distance weighted interpolation
        values = self.values[indices]
        weights = 1 / np.clip(distances, 1e-6, float("inf"))

        if (values.ndim == 3) and (self.array_valued):
            # faster shortcut for large S1/S2 maps, avoids caching by direct summation
            result[valid] = np.einsum(
                "ijk, ij->ik", values, weights / weights.sum(axis=-1)[:, np.newaxis]
            )
        else:
            if self.array_valued:
                weights = np.repeat(weights, self.n_dim).reshape(values.shape)
            result[valid] = np.average(
                values, weights=weights, axis=-2 if self.array_valued else -1
            )

        return result


@export
class InterpolatingMap:
    """Correction map that computes values using inverse-weighted distance interpolation.

    The map must be specified as a json translating to a dictionary like this:
        'coordinate_system' :   [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...],
        'map' :                 [value1, value2, value3, value4, ...]
        'another_map' :         idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are, who you are, etc',
        'timestamp':            unix epoch seconds timestamp

    with the straightforward generalization to 1d and 3d.

    Alternatively, a grid coordinate system can be specified as follows:
        'coordinate_system' :   [['x', [x_min, x_max, n_x]], [['y', [y_min, y_max, n_y]]

    Alternatively, an N-vector-valued map can be specified by an array with
    last dimension N in 'map'.

    The default map name is 'map', I'd recommend you use that.

    For a 0d placeholder map, use
        'points': [],
        'map': 42,
        etc

    Default method return inverse-distance weighted average of nearby 2 * dim points
    Extra support includes RectBivariateSpline, RegularGridInterpolator in scipy
    by pass keyword argument like
        method='RectBivariateSpline'

    The interpolators are called with
        'positions' :  [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
        'map_name'  :  key to switch to map interpolator other than the default 'map'

    """

    metadata_field_names = [
        "timestamp",
        "description",
        "coordinate_system",
        "name",
        "irregular",
        "compressed",
        "quantized",
    ]

    def __init__(self, data, method="WeightedNearestNeighbors", **kwargs):
        if isinstance(data, bytes):
            data = gzip.decompress(data).decode()
        if isinstance(data, (str, bytes)):
            data = json.loads(data)
        assert isinstance(data, dict), f"Expected dictionary data, got {type(data)}"
        self.data = data

        # Decompress / dequantize the map
        # We should support multiple map names!
        if "compressed" in self.data:
            compressor, dtype, shape = self.data["compressed"]
            self.data["map"] = np.frombuffer(
                strax.io.COMPRESSORS[compressor]["decompress"](self.data["map"]), dtype=dtype
            ).reshape(*shape)
            del self.data["compressed"]
        if "quantized" in self.data:
            self.data["map"] = self.data["quantized"] * self.data["map"].astype(np.float32)
            del self.data["quantized"]

        csys = self.data["coordinate_system"]
        if not len(csys):
            self.dimensions = 0
        elif isinstance(csys[0][0], str):
            # Support for specifying coordinate system as a gridspec
            grid = [np.linspace(left, right, points) for _, (left, right, points) in csys]
            csys = np.array(np.meshgrid(*grid, indexing="ij"))
            axes = np.roll(np.arange(len(grid) + 1), -1)
            csys = np.transpose(csys, axes)
            csys = np.array(csys).reshape((-1, len(grid)))
            self.dimensions = len(grid)
        else:
            csys = np.array(csys)
            self.dimensions = len(csys[0])

        self.coordinate_system = csys
        self.interpolators = {}
        self.map_names = sorted([k for k in self.data.keys() if k not in self.metadata_field_names])

        log = logging.getLogger("InterpolatingMap")
        log.debug("Map name: %s" % self.data.get("name", "NO NAME?!"))
        log.debug(
            "Map description:\n    "
            + re.sub(r"\n", r"\n    ", self.data.get("description", "NO DESCRIPTION?!"))
        )
        log.debug("Map names found: %s" % self.map_names)

        for map_name in self.map_names:
            # Specify dtype float to set Nones to nan
            map_data = np.array(self.data[map_name], dtype=np.float64)

            if self.dimensions != 0:
                if len(self.coordinate_system) == len(map_data):
                    array_valued = len(map_data.shape) == 2
                else:
                    array_valued = len(map_data.shape) == self.dimensions + 1

            if self.dimensions == 0:
                # 0 D -- placeholder maps which take no arguments
                # and always return a single value
                def itp_fun(positions):
                    return np.array([map_data])

            elif method == "RectBivariateSpline":
                itp_fun = self._rect_bivariate_spline(csys, map_data, array_valued, **kwargs)

            elif method == "RegularGridInterpolator":
                itp_fun = self._regular_grid_interpolator(csys, map_data, array_valued, **kwargs)

            elif method == "WeightedNearestNeighbors":
                itp_fun = self._weighted_nearest_neighbors(csys, map_data, array_valued, **kwargs)

            else:
                raise ValueError(f"Interpolation method {method} is not supported")

            self.interpolators[map_name] = itp_fun

    def __call__(self, *args, map_name="map"):
        """Returns the value of the map at the position given by coordinates.

        :param positions: array (n_dim) or (n_points, n_dim) of positions
        :param map_name: Name of the map to use.  Default is 'map'.

        """
        return self.interpolators[map_name](*args)

    @staticmethod
    def _rect_bivariate_spline(csys, map_data, array_valued, **kwargs):
        dimensions = len(csys[0])
        grid = [np.unique(csys[:, i]) for i in range(dimensions)]
        grid_shape = [len(g) for g in grid]

        assert dimensions == 2, "RectBivariateSpline interpolate maps of dimension 2"
        assert not array_valued, "RectBivariateSpline does not support interpolating array values"
        map_data = map_data.reshape(*grid_shape)
        kwargs = straxen.filter_kwargs(RectBivariateSpline, kwargs)
        rbs = RectBivariateSpline(grid[0], grid[1], map_data, **kwargs)

        def arg_formated_rbs(positions):
            if isinstance(positions, list):
                positions = np.array(positions)
            return rbs.ev(positions[:, 0], positions[:, 1])

        return arg_formated_rbs

    @staticmethod
    def _regular_grid_interpolator(csys, map_data, array_valued, **kwargs):
        dimensions = len(csys[0])
        grid = [np.unique(csys[:, i]) for i in range(dimensions)]
        grid_shape = [len(g) for g in grid]

        if array_valued:
            map_data = map_data.reshape((*grid_shape, map_data.shape[-1]))
        else:
            map_data = map_data.reshape(*grid_shape)

        config = dict(bounds_error=False, fill_value=None)
        kwargs = straxen.filter_kwargs(RegularGridInterpolator, kwargs)
        config.update(kwargs)

        return RegularGridInterpolator(tuple(grid), map_data, **config)

    @staticmethod
    def _weighted_nearest_neighbors(csys, map_data, array_valued, **kwargs):
        if array_valued:
            map_data = map_data.reshape((-1, map_data.shape[-1]))
        else:
            map_data = map_data.flatten()
        kwargs = straxen.filter_kwargs(InterpolateAndExtrapolate, kwargs)
        return InterpolateAndExtrapolate(csys, map_data, array_valued=array_valued, **kwargs)

    def scale_coordinates(self, scaling_factor, map_name="map"):
        """Scales the coordinate system by the specified factor.

        :params scaling_factor: array (n_dim) of scaling factors if different or single scalar.

        """
        if self.dimensions == 0:
            return
        if hasattr(scaling_factor, "__len__"):
            assert len(scaling_factor) == self.dimensions, (
                f"Scaling factor array dimension {len(scaling_factor)} "
                f"does not match grid dimension {self.dimensions}"
            )
            self._sf = scaling_factor
        if isinstance(scaling_factor, (int, float)):
            self._sf = [scaling_factor] * self.dimensions

        alt_csys = self.coordinate_system
        for i, gp in enumerate(self.coordinate_system):
            alt_csys[i] = [gc * k for (gc, k) in zip(gp, self._sf)]

        map_data = np.array(self.data[map_name])
        if len(self.coordinate_system) == len(map_data):
            array_valued = len(map_data.shape) == 2
        else:
            array_valued = len(map_data.shape) == self.dimensions + 1
        if array_valued:
            map_data = map_data.reshape((-1, map_data.shape[-1]))
        itp_fun = InterpolateAndExtrapolate(
            points=np.array(alt_csys), values=np.array(map_data), array_valued=array_valued
        )
        self.interpolators[map_name] = itp_fun


@export
def save_interpolation_formatted_map(
    itp_map,
    coordinate_system: List,
    filename: str,
    map_name: Optional[str] = None,
    quantum: Optional[float] = None,
    quantum_dtype=np.int16,
    map_description: str = "",
    compressor: Literal["bz2", "zstd", "blosc", "lz4"] = "zstd",
):
    """Make a straxen-style InterpolatingMap.

    To fit the large XENONnT per-PMT maps into strax_auxiliary files,
    quantized them to values of 1e-5,
    and store the maps as 16-bit integer multiples of 1e-5, instead of 64-bit floats.
    :param itp_map: numpy itp_map or list of floats,
    should follow the shape indicated by coordinate_system
    :param coordinate_system: coordinate system of the itp_map,
    list of [['x', [x_min, x_max, n_x]], [['y', [y_min, y_max, n_y], ...] for each dimension,
    or [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
    :param filename: filename with '.pkl' extension
    :param map_name: name of map
    :param quantum: quantum of the map if quantized
    :param map_description: map's description
    :param compressor: key of compressor in strax.io.COMPRESSORS

    """
    if isinstance(itp_map, list):
        itp_map = np.array(itp_map)

    itp_map_shape: Union[list, int]
    coordinate_shape: Union[list, int]
    if isinstance(coordinate_system[0][0], str):
        itp_map_shape = list(itp_map.shape)
        coordinate_shape = [c[1][2] for c in coordinate_system]
        mask = len(itp_map_shape) == len(coordinate_shape)
        mask &= all([hs == cs for hs, cs in zip(itp_map_shape, coordinate_shape)])
    else:
        itp_map_shape = len(itp_map)
        coordinate_shape = len(coordinate_system)
        mask = itp_map_shape == coordinate_shape
    if not mask:
        raise ValueError(
            f"The shape of itp_map: {itp_map_shape} and "
            f"coordinate system: {coordinate_shape} do not match"
        )

    q: Union[int, float]
    if quantum is None:
        # if quantum is not specified, just use float32
        q = 1
        quantum_dtype = np.float32
        map = itp_map.astype(quantum_dtype)
    else:
        q = quantum
        if not np.issubdtype(quantum_dtype, np.integer):
            raise ValueError(
                "If using quantization, quantum_dtype must be an integer type,"
                f" but it is now {quantum_dtype}"
            )
        encode_until = np.iinfo(quantum_dtype).max * q
        if itp_map.max() > encode_until:
            raise ValueError(
                f"Map maximum value is {itp_map.max():.4f},"
                " can encode values until {encode_until:.4f}"
            )
        map = np.round(itp_map / q).astype(quantum_dtype)

    output = dict(
        coordinate_system=coordinate_system,
        map=strax.io.COMPRESSORS[compressor]["compress"](map),
        description=dedent(map_description),
        timestamp=time.time(),
        compressed=(compressor, quantum_dtype, map.shape),
    )
    if quantum is not None:
        output["quantized"] = q
    if map_name is not None:
        output["name"] = map_name

    if "pkl" not in filename:
        warnings.warn("Better use .pkl or .pkl.gz extension for map files")
    splitext = os.path.splitext(filename)
    opener: Callable
    if splitext[-1] == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(filename, mode="wb") as f:
        pickle.dump(output, f)
    print(f"Wrote new map file {filename}, {os.path.getsize(filename) / 1e6:.2f} MB")
