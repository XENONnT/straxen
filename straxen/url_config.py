import json
import typing
from typing import Container
from uuid import uuid4
import numpy as np
import strax
import fsspec
import pandas as pd
import straxen
import inspect
from urllib.parse import urlparse, parse_qs
from ast import literal_eval
from strax.config import OMITTED
import os
import tempfile
import tarfile
import pytz
from utilix import xent_collection
from scipy.interpolate import interp1d

export, __all__ = strax.exporter()

_CACHES = {}


@export
def clear_config_caches():
    for cache in _CACHES.values():
        cache.clear()


@export
def config_cache_size_mb():
    return straxen.total_size(_CACHES)//1e6


def parse_val(val: str):
    '''Attempt to parse a string value as
    a python literal, falls back to returning just
    the original string if cant be parsed.
    '''
    try:
        val = literal_eval(val)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return val


@export
class URLConfig(strax.Config):
    """Dispatch on URL protocol.
    unrecognized protocol returns identity
    inspired by dasks Dispatch and fsspec fs protocols.
    """
    _LOOKUP = {}
    SCHEME_SEP = '://'
    QUERY_SEP = '?'
    PLUGIN_ATTR_PREFIX = 'plugin.'

    def __init__(self, cache=0, **kwargs):
        """
        :param cache: number of values to keep in cache, 
                      if set to True will cache all values
        :param **kwargs: additional keyword arguments accepted by strax.Option
        """
        self.final_type = OMITTED
        super().__init__(**kwargs)
        # Ensure backwards compatibility with Option validation
        # type of the config value can be different from the fetched value.
        if self.type is not OMITTED:
            self.final_type = self.type
            self.type = OMITTED  # do not enforce type on the URL
        if cache:
            cache_len = 100 if cache is True else int(cache) 
            cache = straxen.CacheDict(cache_len=cache_len)
            _CACHES[id(self)] = cache

    @property
    def cache(self):
        return _CACHES.get(id(self), {})

    @classmethod
    def register(cls, protocol, func=None):
        """Register dispatch of `func` on urls
        starting with protocol name `protocol` """

        def wrapper(func):
            if isinstance(protocol, tuple):
                for t in protocol:
                    cls.register(t, func)
                return func

            if not isinstance(protocol, str):
                raise ValueError('Protocol name must be a string.')

            if protocol in cls._LOOKUP:
                raise ValueError(f'Protocol with name {protocol} already registered.')
            cls._LOOKUP[protocol] = func
            return func
        return wrapper(func) if func is not None else wrapper

    def dispatch(self, url, *args, **kwargs):
        """
        Call the corresponding method based on protocol in url.
        chained protocols will be called with the result of the
        previous protocol as input
        overrides are passed to any protocol whos signature can accept them.
        """

        # separate the protocol name from the path
        protocol, _, path = url.partition(self.SCHEME_SEP)

        # find the corresponding protocol method
        meth = self._LOOKUP.get(protocol, None)
        if meth is None:
            # unrecognized protocol
            # evaluate as string-literal
            return url

        if self.SCHEME_SEP in path:
            # url contains a nested protocol
            # first call sub-protocol
            arg = self.dispatch(path, **kwargs)
        else:
            # we are at the end of the chain
            # method should be called with path as argument
            arg = path

        # filter kwargs to pass only the kwargs
        #  accepted by the method.
        kwargs = straxen.filter_kwargs(meth, kwargs)

        return meth(arg, *args, **kwargs)

    @classmethod
    def split_url_kwargs(cls, url):
        """split a url into path and kwargs
        """
        path, _, _ = url.partition(cls.QUERY_SEP)
        kwargs = {}
        for k, v in parse_qs(urlparse(url).query).items():
            # values of query arguments are evaluated as lists
            # split logic depending on length
            n = len(v)
            if not n:
                kwargs[k] = None
            elif n == 1:
                kwargs[k] = parse_val(v[0])
            else:
                kwargs[k] = list(map(parse_val, v))
        return path, kwargs

    @classmethod
    def kwarg_from_url(cls, url: str, key: str):
        path, kwargs = cls.split_url_kwargs(url)
        return kwargs.get(key, None)
        
    @classmethod
    def format_url_kwargs(cls, url, **kwargs):
        '''Add keyword arguments to a URL.
        Sorts all arguments by key for hash consistency
        '''
        url, extra_kwargs = cls.split_url_kwargs(url)
        kwargs = dict(extra_kwargs, **kwargs)
        arg_list = []
        for k, v in sorted(kwargs.items()):
            if isinstance(v, list):
                # lists are passed as multiple arguments with the same key
                arg_list.extend([f"{k}={vi}" for vi in v])
            else:
                arg_list.append(f"{k}={v}")
        arg_str = "&".join(arg_list)
        arg_str = cls.QUERY_SEP + arg_str if arg_str else ''
        return url + arg_str

    def fetch_attribute(self, plugin, value):
        if isinstance(value, str) and value.startswith(self.PLUGIN_ATTR_PREFIX):
            # kwarg is referring to a plugin attribute, lets fetch it
            return getattr(plugin, value[len(self.PLUGIN_ATTR_PREFIX):], value)

        if isinstance(value, list):
            return [self.fetch_attribute(plugin, v) for v in value]

        # kwarg is a literal, add its value to the kwargs dict
        return value

    def fetch(self, plugin):
        """override the Config.fetch method
        this is called when the attribute is accessed
        """
        # first fetch the user-set value 
        # from the config dictionary
        url = super().fetch(plugin)

        if not isinstance(url, str):
            # if the value is not a string it is evaluated
            # as a literal config and returned as is.
            return url

        if self.SCHEME_SEP not in url:
            # no protocol in the url so its evaluated 
            # as string-literal config and returned as is
            return url

        # separate out the query part of the URL which
        # will become the method kwargs
        url, url_kwargs = self.split_url_kwargs(url)


        kwargs = {k: self.fetch_attribute(plugin, v)
                  for k, v in url_kwargs.items()}

        key = None
        value = None
        try:
            # construct a deterministic hash key
            key = strax.deterministic_hash((url, kwargs))
                    
            # fetch from cache if exists
            value = self.cache.get(key, None)
        except TypeError:
            # kwargs contains unhashable types
            # so we cannot cache the result
            pass

        # not in cache, lets fetch it
        if value is None:
            value = self.dispatch(url, **kwargs)

            if key is not None:
                self.cache[key] = value

        return value

    @classmethod
    def protocol_descr(cls):
        rows = []
        for k, v in cls._LOOKUP.items():
            row = {
                'name': f"{k}://",
                'description': v.__doc__,
                'signature': str(inspect.signature(v)),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def print_protocols(cls):
        df = cls.protocol_descr()
        print(df)

    @classmethod
    def evaluate_dry(cls, url: str, **kwargs):
        """
        Utility function to quickly test and evaluate URL configs,
        without the initialization of plugins (so no plugin attributes).
        plugin attributes can be passed as keyword arguments.

        example::

            from straxen import URLConfig
            url_string='cmt://electron_drift_velocity?run_id=027000&version=v3'
            URLConfig.evaluate_dry(url_string)

            # or similarly
            url_string='cmt://electron_drift_velocity?run_id=plugin.run_id&version=v3'
            URLConfig.evaluate_dry(url_string, run_id='027000')

        Please note that this has to be done outside of the plugin, so any
        attributes of the plugin are not yet note to this dry evaluation
        of the url-string.

        :param url: URL to evaluate, see above for example.
        :keyword: any additional kwargs are passed to self.dispatch (see example)
        :return: evaluated value of the URL.
        """
        url_arg, url_kwarg = cls.split_url_kwargs(url)

        combined_kwargs = dict(url_kwarg, **kwargs)
        for k,v in combined_kwargs.items():
            if isinstance(v, str) and cls.PLUGIN_ATTR_PREFIX in v:
                raise ValueError(f'The URL parameter {k} depends on the plugin'
                                'You must specify the value for this parameter'
                                'for this URL to be evaluated correctly.'
                                f'Try passing {k} as a keyword argument.'
                                f'e.g.: `URLConfig.evaluate_dry({url}, {k}=SOME_VALUE)`')

        return cls.dispatch(cls, url_arg, **combined_kwargs)

@URLConfig.register('cmt')
def get_correction(name: str,
                   run_id: str = None,
                   version: str = 'ONLINE',
                   detector: str = 'nt',
                   **kwargs):
    """Get value for name from CMT"""

    if run_id is None:
        raise ValueError('Attempting to fetch a correction without a run id.')    

    return straxen.get_correction_from_cmt(run_id, (name, version, detector == 'nt'))


@URLConfig.register('resource')
def get_resource(name: str,
                 fmt: str = 'text',
                 **kwargs):
    """
    Fetch a straxen resource

    Allow a direct download using <fmt='abs_path'> otherwise kwargs are
    passed directly to straxen.get_resource.
    """
    if fmt == 'abs_path':
        downloader = straxen.MongoDownloader()
        return downloader.download_single(name)
    return straxen.get_resource(name, fmt=fmt)


@URLConfig.register('fsspec')
def read_file(path: str, **kwargs):
    """Support fetching files from arbitrary filesystems
    """
    with fsspec.open(path, **kwargs) as f:
        content = f.read()
    return content


@URLConfig.register('json')
def read_json(content: str, **kwargs):
    """Load json string as a python object
    """
    return json.loads(content)


@URLConfig.register('take')
def get_key(container: Container, take=None, **kwargs):
    """ return a single element of a container
    """
    if take is None:
        return container
    if not isinstance(take, list):
        take = [take]

    # support for multiple keys for
    # nested objects
    for t in take:
        container = container[t]

    return container


@URLConfig.register('format')
def format_arg(arg: str, **kwargs):
    """apply pythons builtin format function to a string"""
    return arg.format(**kwargs)


@URLConfig.register('itp_map')
def load_map(some_map, method='WeightedNearestNeighbors', **kwargs):
    """Make an InterpolatingMap"""
    return straxen.InterpolatingMap(some_map, method=method, **kwargs)


@URLConfig.register('bodega')
def load_value(name: str, bodega_version=None):
    """Load a number from BODEGA file"""
    if bodega_version is None:
        raise ValueError('Provide version see e.g. tests/test_url_config.py')
    nt_numbers = straxen.get_resource("XENONnT_numbers.json", fmt="json")
    return nt_numbers[name][bodega_version]["value"]


@URLConfig.register('tf')
def open_neural_net(model_path: str, custom_objects=None, **kwargs):
    # Nested import to reduce loading time of import straxen and it not
    # base requirement
    import tensorflow as tf
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No file at {model_path}')
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar = tarfile.open(model_path, mode="r:gz")
        tar.extractall(path=tmpdirname)
        return tf.keras.models.load_model(tmpdirname, custom_objects=custom_objects)


@URLConfig.register('itp_dict')
def get_itp_dict(loaded_json,
                 run_id=None,
                 time_key='time',
                 itp_keys='correction',
                 **kwargs) -> typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]:
    """
    Interpolate a dictionary at the start time that is queried from
    a run-id.

    :param loaded_json: a dictionary with a time-series
    :param run_id: run_id
    :param time_key: key that gives the timestamps
    :param itp_keys: which keys from the dict to read. Should be
        comma (',') separated!

    :return: Interpolated values of dict at the start time, either
        returned as an np.ndarray (single value) or as a dict
        (multiple itp_dict_keys)
    """
    keys = strax.to_str_tuple(itp_keys.split(','))
    for key in list(keys) + [time_key]:
        if key not in loaded_json:
            raise KeyError(f"The json does contain the key '{key}'. Try one of: {loaded_json.keys()}")

    times = loaded_json[time_key]

    # get start time of this run. Need to make tz-aware
    start = xent_collection().find_one({'number': int(run_id)}, {'start': 1})['start']
    start = pytz.utc.localize(start).timestamp() * 1e9

    try:
        if len(strax.to_str_tuple(keys)) > 1:
            return {key:
                    interp1d(times, loaded_json[key], bounds_error=True)(start)
                    for key in keys}

        else:
            interp = interp1d(times, loaded_json[keys[0]], bounds_error=True)
            return interp(start)
    except ValueError as e:
        raise ValueError(f"Correction is not defined for run {run_id}") from e


@URLConfig.register('rekey_dict')
def rekey_dict(d, replace_keys='', with_keys=''):
    '''
    :param d: dictionary that will have its keys renamed
    :param replace_keys: comma-separated string of keys that will be replaced
    :param with_keys:  comma-separated string of keys that will replace the replace_keys
    :return: dictionary with renamed keys
    '''
    new_dict = d.copy()
    replace_keys = strax.to_str_tuple(replace_keys.split(','))
    with_keys = strax.to_str_tuple(with_keys.split(','))
    if len(replace_keys) != len(with_keys):
        raise RuntimeError("replace_keys and with_keys must have the same length")
    for old_key, new_key in zip(replace_keys, with_keys):
        new_dict[new_key] = new_dict.pop(old_key)
    return new_dict


@URLConfig.register('legacy-to-pe')
def get_fixed_pe(name: str):
    """Return a fixed value for a given name"""
    from .get_corrections import FIXED_TO_PE

    return FIXED_TO_PE[name]


@URLConfig.register('legacy-thresholds')
def get_thresholds(model: str):
    """Return a fixed value for a given model"""
    
    return straxen.hit_min_amplitude(model)
