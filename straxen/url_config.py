import os
import json
import pytz
import strax
import typing
import fsspec
import numbers
import straxen
import inspect
import tarfile
import tempfile

import numpy as np
import pandas as pd

from ast import literal_eval
from strax.config import OMITTED
from utilix import xent_collection
from scipy.interpolate import interp1d
from straxen.misc import filter_kwargs
from urllib.parse import urlparse, parse_qs
from typing import Container, Mapping, Union

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
    _PREPROCESSORS = ()

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

    @classmethod
    def register_preprocessor(cls, func=None, precedence=0):
        '''Register a new processor to modify the config values
        before they are used.
        '''
        def wrapper(func):
            entry = (precedence, func)
            if entry in cls._PREPROCESSORS:
                raise ValueError(f'This processor is already registered.')
            cls._PREPROCESSORS += (entry, )
            return func
        return wrapper(func) if func is not None else wrapper

    @classmethod
    def eval(cls, protocol: str,
                  arg: Union[str,tuple] = None,
                  kwargs: dict = None):
        '''Evaluate a URL/AST by recusively dispatching protocols by name 
            with argument arg and keyword arguments kwargs
           and return the value. If protocol does not exist, returnes arg
        :param protocol: name of the protocol or a URL
        :param arg: argument to pass to protocol, can be another (sub-protocol,
            arg, kwargs) tuple, in which case sub-protocol will be evaluated
            and passed to protocol
        :param kwargs: keyword arguments to be passed to the protocol
        :return: (Any) The return value of the protocol on these arguments
        '''
        

        if protocol is not None and arg is None:
            protocol, arg, kwargs = cls.url_to_ast(protocol)

        if protocol is None:
            return arg

        if kwargs is None:
            kwargs = {}

        meth = cls._LOOKUP[protocol]

        if isinstance(arg, tuple):
            arg = cls.eval(*arg)
        
        # Just to be on the safe side
        kwargs = straxen.filter_kwargs(meth, kwargs)

        return meth(arg, **kwargs)

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

    @classmethod
    def lookup_value(cls, value, **namespace):
        '''Optionally fetch an attribute from namespace
        if value is a string with cls.NAMESPACE_SEP in
        it, the string is split and the first part is used
        to lookup an object in namespace and the second part
        is used to lookup the value in the object.
        If the value is not a string or the target object is
        not in the namesapce, the value is returned as is.
        '''

        if isinstance(value, list):
            return [cls.lookup_value(v, **namespace) for v in value]

        if isinstance(value, str) and '.' in value:
            name, _, key = value.partition('.')
            if name in namespace:
                obj = namespace[name]
                if isinstance(obj, Mapping):
                    value = obj.get(key, value)
                else:
                    value = getattr(obj, key, value)
                
        return value

    def validate(self, config,
                 run_id=None,   # TODO: will soon be removed
                 run_defaults=None, set_defaults=True):
        """This method is called by the context on plugin initialization
        at this stage, the run_id and context config are already known but the
        config values are not yet set on the plugin. Therefore its the perfect
        place to run any preprocessors on the config values to make any needed
        changes before the configs are hashed.
        """
        super().validate(config, run_id, run_defaults, set_defaults)

        cfg = config[self.name]

        if not isinstance(cfg, str) or self.SCHEME_SEP not in cfg:
            # if the value is not a url config it is validated against
            # its intended type (final_type)
            if self.final_type is not OMITTED and not isinstance(cfg, self.final_type):
                # TODO replace back with InvalidConfiguration
                UserWarning(
                    f"Invalid type for option {self.name}. "
                    f"Excepted a {self.final_type}, got a {type(cfg)}")
            return

        sorted_preprocessors = reversed(sorted(self._PREPROCESSORS,
                                               key=lambda x: x[0]))

        full_kwargs = dict(run_id=run_id, 
                           run_defaults=run_defaults, 
                           set_defaults=set_defaults)

        for _, preprocessor in sorted_preprocessors:
            kwargs = filter_kwargs(preprocessor, full_kwargs)
            cfg = preprocessor(cfg, **kwargs)
        
        config[self.name] = cfg

    def fetch(self, plugin):
        """override the Config.fetch method
           this is called when the attribute is accessed
           from withing the Plugin instance
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
        url, kwargs = self.split_url_kwargs(url)
        
        # resolve any referenced to plugin attributes
        kwargs = {k: self.lookup_value(v, config=plugin.config, plugin=plugin)
                  for k, v in kwargs.items()}

        protocol, arg, kwargs = self.url_to_ast(url, **kwargs)

        # construct a deterministic hash key
        key = strax.deterministic_hash((protocol, arg, kwargs))

        # fetch from cache if exists
        value = self.cache.get(key, None)

        # not in cache, lets fetch it
        if value is None:
            value = self.eval(protocol, arg, kwargs)
            self.cache[key] = value

        return value

    @classmethod
    def format_url_kwargs(cls, url, **kwargs):
        """Add keyword arguments to a URL.
        Sorts all arguments by key for hash consistency
        """
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

    @classmethod
    def ast_to_url(cls,
                   protocol: Union[str, tuple],
                   arg: Union[str, tuple] = None,
                   kwargs: dict = None):
        """Convert a protocol abstract syntax tree to a valid URL 
        """

        if isinstance(protocol, tuple):
            protocol, arg, kwargs = protocol

        if kwargs is None:
            kwargs = {}

        if protocol is None:
            return arg
        
        if isinstance(arg, (list, dict, numbers.Number)) and protocol != 'json':
            arg = ('json', json.dumps(arg),)

        if isinstance(arg, tuple):
            arg = cls.ast_to_url(*arg)

        if not isinstance(arg, str):
            raise TypeError(f"Type {type(arg)} is not supported as an argument.")

        arg, extra_kwargs = cls.split_url_kwargs(arg)

        kwargs.update(extra_kwargs)
        
        url = f'{protocol}{cls.SCHEME_SEP}{arg}'
        
        url = cls.format_url_kwargs(url, **kwargs)

        return url

    @classmethod
    def url_to_ast(cls, url, **kwargs):
        """Convert a URL to a protocol abstract syntax tree
        """
        if not isinstance(url, str):
            raise TypeError(f'URL must be a string, got {type(url)}')

        if cls.SCHEME_SEP not in url:
            # no protocol in the url so its evaluated
            # as string-literal config and returned as is
            return None, url, {}

        # separate the protocol name from the path
        protocol, _, path = url.partition(cls.SCHEME_SEP)

        # find the corresponding protocol method
        meth = cls._LOOKUP.get(protocol, None)
        if meth is None:
            # unrecognized protocol
            # evaluate as string-literal
            return None, url, {}
        
        arg, url_kwargs = cls.split_url_kwargs(path)
        kwargs.update(url_kwargs)


        if cls.SCHEME_SEP in arg:
            # url contains a nested protocol
            # first parsce sub-protocol
            arg = cls.url_to_ast(arg, **kwargs)

        # Filter unused kwargs for this method.
        # This is done also at the eval level but
        # probably better to be safe.
        kwargs = straxen.filter_kwargs(meth, kwargs)
        
        # Always sort kwargs for consistent ASTs
        kwargs = dict(sorted(kwargs.items()))

        return protocol, arg, kwargs

    @classmethod
    def are_equal(cls, first, second):
        """Return whether two URLs are equivalent (have equal ASTs)
        """
        return cls.url_to_ast(first) == cls.url_to_ast(second)

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
        protocol, url_arg, url_kwarg = cls.url_to_ast(url)

        combined_kwargs = dict(url_kwarg, **kwargs)

        for k,v in combined_kwargs.items():
            if isinstance(v, str) and cls.PLUGIN_ATTR_PREFIX in v:
                raise ValueError(f'The URL parameter {k} depends on the plugin'
                                'You must specify the value for this parameter'
                                'for this URL to be evaluated correctly.'
                                f'Try passing {k} as a keyword argument.'
                                f'e.g.: `URLConfig.evaluate_dry({url}, {k}=SOME_VALUE)`')

        return cls.eval(protocol, url_arg, combined_kwargs)

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
def load_map(some_map, method='WeightedNearestNeighbors', scale_coordinates=None, **kwargs):
    """Make an InterpolatingMap"""
    itp_map = straxen.InterpolatingMap(some_map, method=method, **kwargs)
    if scale_coordinates is not None:
        itp_map.scale_coordinates(scale_coordinates)
    return itp_map

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
