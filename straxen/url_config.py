import json
from typing import Container, Union
import strax
import fsspec
import pandas as pd
import straxen
import inspect
import numbers
from urllib.parse import urlparse, parse_qs
from ast import literal_eval
from strax.config import OMITTED
import os
import tempfile
import tarfile
from functools import lru_cache
export, __all__ = strax.exporter()

_CACHES = {}


@export
def clear_config_caches():
    for cache in _CACHES.values():
        cache.clear()


@export
def config_cache_size_mb():
    return straxen.total_size(_CACHES)//1e6


def parse_val(val):
    try:
        val = literal_eval(val)
    except ValueError:
        pass
    return val


class URLConfigError(Exception):
    pass


class PreProcessorError(URLConfigError):
    pass


@export
class URLConfig(strax.Config):
    """Dispatch on URL protocol.
    unrecognized protocol returns identity
    inspired by dasks Dispatch and fsspec fs protocols.
    """
    _PLUGIN_CLASS = None
    _LOOKUP = {}
    _PREPROCESSORS = {}
    SCHEME_SEP = '://'
    QUERY_SEP = '?'
    PLUGIN_ATTR_PREFIX = 'plugin.'
    CONFIG_PREFIX = 'config.'

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
            # do not enforce type on the URL
            self.type = OMITTED
        if cache:
            cache_len = 100 if cache is True else int(cache)
            cache = straxen.CacheDict(cache_len=cache_len)
            _CACHES[id(self)] = cache

    @property
    def cache(self):
        return _CACHES.get(id(self), {})

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self._PLUGIN_CLASS = owner

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

            if not len(inspect.signature(func).parameters):
                raise TypeError(f"Cant use {func.__name__} for protocol {protocol}.\
                     Protocols must accept at least one positional argument.")

            if protocol in cls._LOOKUP:
                raise ValueError(f'Protocol with name {protocol} already registered.')
            cls._LOOKUP[protocol] = func
            return func
        return wrapper(func) if func is not None else wrapper

    @classmethod
    def register_preprocessor(cls, protocol, func=None):
        """Register dispatch of `func` as a preprocessor on urls
        starting with protocol name `protocol` """

        def wrapper(func):
            if isinstance(protocol, tuple):
                for t in protocol:
                    cls.register(t, func)
                return func

            if not isinstance(protocol, str):
                raise ValueError('Protocol name must be a string.')

            if protocol not in cls._LOOKUP:
                ValueError(f'No protocol {protocol} registered.\
                 Can only register preprocessors for existing protocols.')

            if protocol in cls._PREPROCESSORS:
                raise ValueError(f'Preprocessor for protocol {protocol}\
                     already registered.')

            # FIXME: maybe check here that signature of preprocessor
            # matches (url, arg, **kwargs)

            cls._PREPROCESSORS[protocol] = func
            return func
        return wrapper(func) if func is not None else wrapper

    @classmethod
    def dispatch_protocol(cls, protocol: str,
                        arg: Union[str,tuple] = None,
                        kwargs: dict = None):

        if arg is None:
            protocol, arg, kwargs = cls.parse_url(protocol)
            
        if kwargs is None:
            kwargs = {}

        if protocol is None:
            return arg

        meth = cls._LOOKUP[protocol]

        if isinstance(arg, tuple):
            arg = cls.dispatch_protocol(*arg)
        
        # Just to be on the safe side
        kwargs = cls.filter_kwargs(meth, kwargs)

        return meth(arg, **kwargs)

    @classmethod
    def dispatch_preprocessor(cls, protocol: str,
                            arg: Union[str,tuple] = None,
                            kwargs: dict = None):
        if arg is None:
            protocol, arg, kwargs = cls.parse_url(protocol)
            
        kwargs_overrides = {}

        if isinstance(arg, tuple) and protocol in cls._LOOKUP:
            _, _, kwargs_overrides = cls.dispatch_preprocessor(*arg)

        if protocol not in cls._PREPROCESSORS:
            return protocol, arg, kwargs
        
        meth = cls._PREPROCESSORS[protocol]

        if kwargs is None:
            kwargs = {}

        meth_arg = arg
        if isinstance(meth_arg, tuple):
            meth_arg = cls.dispatch_protocol(*meth_arg)
            
        # Just to be on the safe side
        kwargs = cls.filter_kwargs(meth, kwargs)
        result = meth(meth_arg, **kwargs)
        if isinstance(result, dict):
            kwargs_overrides.update(result)
        elif isinstance(result, str):
            protocol, arg, kwargs_overrides = cls.parse_url(result)
        elif isinstance(result, tuple) and len(result)==3:
            protocol, arg, kwargs_overrides = result
        
        return protocol, arg, kwargs_overrides

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

    @staticmethod
    def filter_kwargs(func, kwargs):
        """Filter out keyword arguments that
            are not in the call signature of func
            and return filtered kwargs dictionary
        """
        params = inspect.signature(func).parameters
        if any([str(p).startswith('**') for p in params.values()]):
            # if func accepts wildcard kwargs, return all
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

    def fetch_attribute(self, plugin, value, **config):

        if isinstance(value, str):
            if value.startswith(self.PLUGIN_ATTR_PREFIX):
                # kwarg is referring to a plugin attribute, lets fetch it
                attr = value[len(self.PLUGIN_ATTR_PREFIX):]
                return getattr(plugin, attr, value)
            elif value.startswith(self.CONFIG_PREFIX):
                key = value[len(self.CONFIG_PREFIX):]
                return config.get(key, value)
            
        if isinstance(value, list):
            return [self.fetch_attribute(plugin, v, **config) for v in value]

        # kwarg is a literal, add its value to the kwargs dict
        return value

    @classmethod
    def format_url_kwargs(cls, url, **kwargs):
        url, extra_kwargs = cls.split_url_kwargs(url)
        kwargs.update(extra_kwargs)
        arg_str = "&".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        arg_str = cls.QUERY_SEP + arg_str if arg_str else ''
        return url + arg_str

    @classmethod
    def ast_to_url(cls, protocol: str, arg: Union[str,tuple], kwargs: dict=None):
        
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
    def parse_url(cls, url, **kwargs):
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

        kwargs = cls.filter_kwargs(meth, kwargs)
        kwargs = dict(sorted(kwargs.items()))

        if cls.SCHEME_SEP in arg:
            # url contains a nested protocol
            # first parsce sub-protocol
            arg = cls.parse_url(arg, **kwargs)
        
        return protocol, arg, kwargs


    def fetch(self, plugin):
        '''override the Config.fetch method
           this is called when the attribute is accessed
        '''
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
        kwargs = {k: self.fetch_attribute(plugin, v, **plugin.config)
                  for k, v in kwargs.items()}

        protocol, arg, kwargs = self.parse_url(url, **kwargs)

        # construct a deterministic hash key
        key = strax.deterministic_hash((protocol, arg, kwargs))

        # fetch from cache if exists
        value = self.cache.get(key, None)

        # not in cache, lets fetch it
        if value is None:
            value = self.dispatch_protocol(protocol, arg, kwargs)
            self.cache[key] = value

        return value

    def validate(self, config,
                 run_id=None,   # TODO: will soon be removed
                 run_defaults=None, set_defaults=True):

        super().validate(config, run_id, run_defaults, set_defaults)

        url = config[self.name]

        if not isinstance(url, str) or self.SCHEME_SEP not in url:
            # if the value is not a url config it is validated against
            # its intended type (final_type)
            if self.final_type is not OMITTED and not isinstance(url, self.final_type):
                # TODO replace back with InvalidConfiguration
                UserWarning(
                    f"Invalid type for option {self.name}. "
                    f"Excepted a {self.final_type}, got a {type(url)}")
            return

        # separate out the query part of the URL which
        # will become the method kwargs
        protocol, arg, kwargs = self.parse_url(url)

        plugin = self._PLUGIN_CLASS()
        plugin.config = {k:v for k,v in config.items() if k in plugin.takes_config}

        # fetch any kwargs that reference other configs
        original_kwargs = kwargs
        kwargs = {k: self.fetch_attribute(plugin, v, **config) for k,v in kwargs.items()}
        
        # dispatch any protocol preprocessors
        protocol, arg, kwargs = self.dispatch_preprocessor(protocol, arg, kwargs)

        # update with any overrides from preprocessors
        original_kwargs.update(kwargs)
        
        # build the modified URL from the preprocessor results
        url = self.ast_to_url(protocol, arg, original_kwargs)

        # finally replace config value with processed url
        config[self.name] = url

    @classmethod
    def protocol_descr(cls):
        '''Return a dataframe with descriptions
        and call signature of all registered protocols
        '''
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

    for t in take:
        container = container[t]

    return container


@URLConfig.register('format')
def format_arg(arg: str, **kwargs):
    """apply pythons builtin format function to a string"""
    return arg.format(**kwargs)


@URLConfig.register_preprocessor('cmt')
def replace_global_version(correction, version=''):
    if version.startswith('global'):
        local_versions = get_cmt_local_versions(version)
        v = local_versions.get(correction, version)
        return dict(version=v)


@lru_cache(maxsize=2)
def get_cmt_local_versions(global_version):
    cmt = straxen.CorrectionsManagementServices()
    return cmt.get_local_versions(global_version)


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
def open_neural_net(model_path: str, **kwargs):
    # Nested import to reduce loading time of import straxen and it not
    # base requirement
    import tensorflow as tf
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No file at {model_path}')
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar = tarfile.open(model_path, mode="r:gz")
        tar.extractall(path=tmpdirname)
        return tf.keras.models.load_model(tmpdirname)
