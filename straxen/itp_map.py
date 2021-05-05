import logging
import gzip
import json
import re

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline, interp2d

import strax
export, __all__ = strax.exporter()


@export
class InterpolateAndExtrapolate:
    """Linearly interpolate- and extrapolate using inverse-distance
    weighted averaging between nearby points.
    """

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
        distances, indices = self.kdtree.query(points, self.neighbours_to_use)

        result = np.ones(len(points)) * float('nan')
        if self.array_valued:
            result = np.repeat(result.reshape(-1, 1), self.n_dim, axis=1)

        # If one of the coordinates is NaN, the neighbour-query fails.
        # If we don't filter these out, it would result in an IndexError
        # as the kdtree returns an invalid index if it can't find neighbours.
        valid = (distances < float('inf')).max(axis=-1)

        values = self.values[indices[valid]]
        weights = 1 / np.clip(distances[valid], 1e-6, float('inf'))
        if self.array_valued:
            weights = np.repeat(weights, self.n_dim).reshape(values.shape)

        result[valid] = np.average(values, weights=weights,
                                   axis=-2 if self.array_valued else -1)
        return result


@export
class InterpolatingMap:
    """Correction map that computes values using inverse-weighted distance
    interpolation.

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
    Extra supports include interp2d, RectBivariateSpline in scipy by pass keyword argument like
        method='interp2d'
    """
    metadata_field_names = ['timestamp', 'description', 'coordinate_system',
                            'name', 'irregular', 'compressed', 'quantized']

    def __init__(self, data, method='WeightedNearestNeighbours', **kwargs):
        if isinstance(data, bytes):
            data = gzip.decompress(data).decode()
        if isinstance(data, (str, bytes)):
            data = json.loads(data)
        assert isinstance(data, dict), f"Expected dictionary data, got {type(data)}"
        self.data = data

        cs = self.data['coordinate_system']
        if not len(cs):
            self.dimensions = 0
        elif isinstance(cs[0], list) and isinstance(cs[0][0], str):
            # Support for specifying coordinate system as a gridspec
            grid = [np.linspace(left, right, points)
                    for _, (left, right, points) in cs]
            cs = np.array(np.meshgrid(*grid, indexing='ij'))
            cs = np.transpose(cs, np.roll(np.arange(len(grid)+1), -1))
            cs = np.array(cs).reshape((-1, len(grid)))
            self.dimensions = len(grid)
        else:
            cs = np.array(cs)
            grid = [np.unique(cs[:, i]) for i in range(len(cs[0]))]
            self.dimensions = len(cs[0])

        self.coordinate_system = cs
        self.interpolators = {}
        self.map_names = sorted([k for k in self.data.keys()
                                 if k not in self.metadata_field_names])

        log = logging.getLogger('InterpolatingMap')
        log.debug('Map name: %s' % self.data['name'])
        log.debug('Map description:\n    ' +
                  re.sub(r'\n', r'\n    ', self.data['description']))
        log.debug("Map names found: %s" % self.map_names)

        for map_name in self.map_names:
            # Decompress / dequantize the map
            # TODO: support multiple map names
            if 'compressed' in self.data:
                compressor, dtype, shape = self.data['compressed']
                self.data[map_name] = np.frombuffer(
                    strax.io.COMPRESSORS[compressor]['decompress'](self.data[map_name]),
                    dtype=dtype).reshape(*shape)
            if 'quantized' in self.data:
                self.data[map_name] = self.data['quantized'] * self.data[map_name].astype(np.float32)

            # Specify dtype float to set Nones to nan
            map_data = np.array(self.data[map_name], dtype=np.float)
            array_valued = len(map_data.shape) == self.dimensions + 1
            if array_valued:
                map_data = map_data.reshape((-1, map_data.shape[-1]))

            if self.dimensions == 0:
                # 0 D -- placeholder maps which take no arguments
                # and always return a single value
                def itp_fun(positions):
                    return np.array([map_data])

            elif method == 'RectBivariateSpline':
                assert self.dimensions == 2, 'interp2d interpolate maps of dimension 2'
                assert not array_valued, 'interp2d does not support interpolating array values'
                itp_fun = RectBivariateSpline(grid[0], grid[1], map_data, **kwargs)

            elif method == 'interp2d':
                assert self.dimensions == 2, 'interp2d interpolate maps of dimension 2'
                assert not array_valued, 'interp2d does not support interpolating array values'
                itp_fun = interp2d(cs[:, 0], cs[:, 1], map_data, **kwargs)

            elif method == 'WeightedNearestNeighbours':
                itp_fun = InterpolateAndExtrapolate(points=np.array(cs),
                                                    values=np.array(map_data),
                                                    array_valued=array_valued)

            else:
                raise ValueError(f'Interpolation method {method} is not supported')

            self.interpolators[map_name] = itp_fun

    def __call__(self, positions, map_name='map'):
        """Returns the value of the map at the position given by coordinates
        :param positions: array (n_dim) or (n_points, n_dim) of positions
        :param map_name: Name of the map to use. Default is 'map'.
        """
        return self.interpolators[map_name](positions)

    def scale_coordinates(self, scaling_factor, map_name='map'):
        """Scales the coordinate system by the specified factor
        :params scaling_factor: array (n_dim) of scaling factors if different or single scalar.
        """
        if self.dimensions == 0:
            return
        if hasattr(scaling_factor, '__len__'):
            assert (len(scaling_factor) == self.dimensions), \
                f"Scaling factor array dimension {len(scaling_factor)} " \
                f"does not match grid dimension {self.dimensions}"
            self._sf = scaling_factor
        if isinstance(scaling_factor, (int, float)):
            self._sf = [scaling_factor] * self.dimensions

        alt_cs = self.coordinate_system
        for i, gp in enumerate(self.coordinate_system):
            alt_cs[i] = [gc * k for (gc, k) in zip(gp, self._sf)]

        map_data = np.array(self.data[map_name])
        array_valued = len(map_data.shape) == self.dimensions + 1
        if array_valued:
            map_data = map_data.reshape((-1, map_data.shape[-1]))
        itp_fun = InterpolateAndExtrapolate(points=np.array(alt_cs),
                                            values=np.array(map_data),
                                            array_valued=array_valued)
        self.interpolators[map_name] = itp_fun
