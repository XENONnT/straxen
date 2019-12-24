import logging
import gzip
import json
import re

import numpy as np
from scipy.spatial import cKDTree

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

    """
    data_field_names = ['timestamp', 'description', 'coordinate_system',
                        'name', 'irregular']

    def __init__(self, data):
        self.log = logging.getLogger('InterpolatingMap')

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
            cs = np.array(np.meshgrid(*grid))
            cs = np.transpose(cs, np.roll(np.arange(len(grid)+1), -1))
            cs = np.array(cs).reshape((-1, len(grid)))
            self.dimensions = len(grid)
        else:
            self.dimensions = len(cs[0])

        self.coordinate_system = cs
        self.interpolators = {}
        self.map_names = sorted([k for k in self.data.keys()
                                 if k not in self.data_field_names])
        self.log.debug('Map name: %s' % self.data['name'])
        self.log.debug('Map description:\n    ' +
                       re.sub(r'\n', r'\n    ', self.data['description']))
        self.log.debug("Map names found: %s" % self.map_names)

        for map_name in self.map_names:
            map_data = np.array(self.data[map_name])
            array_valued = len(map_data.shape) == self.dimensions + 1
            if self.dimensions == 0:
                # 0 D -- placeholder maps which take no arguments
                # and always return a single value
                def itp_fun(positions):
                    return map_data * np.ones_like(positions)
            if array_valued:
                map_data = map_data.reshape((-1, map_data.shape[-1]))
            itp_fun = InterpolateAndExtrapolate(points=np.array(cs),
                                                values=np.array(map_data),
                                                array_valued=array_valued)

            self.interpolators[map_name] = itp_fun

    def __call__(self, positions, map_name='map'):
        """Returns the value of the map at the position given by coordinates
        :param positions: array (n_dim) or (n_points, n_dim) of positions
        :param map_name: Name of the map to use. Default is 'map'.
        """
        return self.interpolators[map_name](positions)
