import json
from typing import Container, Mapping, Union
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


def parse_val(val: str):
    """Attempt to parse a string value as
    a python literal, falls back to returning just
    the original string if cant be parsed.
    """
    try:
        val = literal_eval(val)
    except ValueError:
        pass
    except SyntaxError:
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
    def dispatch_protocol(cls,
                          protocol: str,
                          arg: Union[str, tuple] = None,
                          kwargs: dict = None):
        """dispatch protocol with argument and kwargs
        
        :param protocol: name of the protocol or a URL
        :param arg: argument to pass to protocol, can be another (sub-protocol,
            arg, kwargs) tuple, in which case sub-protocol will be evaluated
            and passed to protocol
        :param kwargs: keyword arguments to be passed to the protocol
        :return: (Any) The return value of the protocol on these arguments
        """

        if arg is None:
            protocol, arg, kwargs = cls.url_to_ast(protocol)
            
        if kwargs is None:
            kwargs = {}

        if protocol is None:
            return arg

        meth = cls._LOOKUP[protocol]

        if isinstance(arg, tuple):
            arg = cls.dispatch_protocol(*arg)
        
        # Just to be on the safe side
        kwargs = straxen.filter_kwargs(meth, kwargs)

        return meth(arg, **kwargs)

    @classmethod
    def dispatch_preprocessor(cls,
                              protocol: str,
                              arg: Union[str, tuple] = None,
                              kwargs: dict = None):
        """dispatch protocol preprocessors

        :param protocol: name of the protocol or a URL
        :param arg: argument to pass to protocol, can be another (sub-protocol,
            arg, kwargs) tuple, in which case any sub-preprocessors will be
            applied if they exist and sub-protocol will be evaluated and passed
             to the preprocessor of `protocol` if it exists
        :param kwargs: keyword arguments to be passed to the protocol
        :return: The modified abstract syntax tree
        """
        if arg is None:
            # Support passing a URL, is converted to an AST
            protocol, arg, kwargs = cls.url_to_ast(protocol)
            
        kwargs_overrides = {}

        if isinstance(arg, tuple) and protocol in cls._LOOKUP:
            # since this is a valid protocol with a sub-protocol, it may have 
            # a preprocessor registered for its sub-protocol
            arg = cls.dispatch_preprocessor(*arg)
        
        if protocol not in cls._PREPROCESSORS:
            # no preprocessor registered for this protocol
            # return any overrides from sub-protocols preprocessor
            return protocol, arg, kwargs_overrides
        
        meth = cls._PREPROCESSORS[protocol]

        if kwargs is None:
            kwargs = {}

        meth_arg = arg
        if isinstance(meth_arg, tuple):
            meth_kwargs = dict(kwargs, **meth_arg[2])

            meth_arg = cls.dispatch_protocol(*meth_arg[:2], meth_kwargs)
            
        # Just to be on the safe side
        kwargs = straxen.filter_kwargs(meth, kwargs)
        result = meth(meth_arg, **kwargs)

        if isinstance(result, dict):
            # a dictionary is interpreted as just kwarg overrides
            kwargs_overrides = result
        elif isinstance(result, str):
            # a string is interpreted as a URL to replace existing one
            protocol, arg, kwargs_overrides = cls.url_to_ast(result)
        elif result is not None:
            # any other non None object is interpreted as a literal value
            protocol, arg = None, result

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

    @classmethod
    def kwarg_from_url(cls, url: str, key: str):
        path, kwargs = cls.split_url_kwargs(url)
        return kwargs.get(key, None)
        
    @staticmethod
    def evaluate(value, **namespace):
        """Fetch an attribute from namespace
        """

        if isinstance(value, list):
            return [URLConfig.evaluate(v, **namespace) for v in value]

        if isinstance(value, str) and '.' in value:
            name, _, key = value.partition('.')
            if name in namespace:
                obj = namespace[name]
                if isinstance(obj, Mapping):
                    value = obj.get(key, value)
                else:
                    value = getattr(obj, key, value)
                
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

        kwargs = straxen.filter_kwargs(meth, kwargs)
        kwargs = dict(sorted(kwargs.items()))

        if cls.SCHEME_SEP in arg:
            # url contains a nested protocol
            # first parsce sub-protocol
            arg = cls.url_to_ast(arg, **kwargs)
        
        return protocol, arg, kwargs

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
        kwargs = {k: self.evaluate(v, config=plugin.config, plugin=plugin)
                  for k, v in kwargs.items()}

        protocol, arg, kwargs = self.url_to_ast(url, **kwargs)

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
        """This method is called by the context on plugin initialization
        at this stage, the run_id and context config are already known but the
        config values are not yet set on the plugin. Therefore its the perfect
        place to run any preprocessors on the config values to make any needed
        changes before the configs are hashed.
        """
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
        protocol, arg, kwargs = self.url_to_ast(url)
        
        plugin = self._PLUGIN_CLASS()
        plugin.run_id = run_id
        plugin.config = {k:v for k,v in config.items() if k in plugin.takes_config}

        # fetch any kwargs that reference other configs
        original_kwargs = kwargs
        kwargs = {k: self.evaluate(v, config=config, plugin=plugin) for k,v in kwargs.items()}
        # dispatch any protocol preprocessors
        protocol, arg, kwargs = self.dispatch_preprocessor(protocol, arg, kwargs)

        # update with any overrides from preprocessors
        original_kwargs.update(kwargs)
        # build the modified URL from the preprocessor results
        url = self.ast_to_url(protocol, arg, original_kwargs)
        # finally replace config value with processed url
        config[self.name] = url

    @classmethod
    def are_equal(cls, first, second):
        """Return whether two URLs are equivalent (have equal ASTs)
        """
        return cls.url_to_ast(first) == cls.url_to_ast(second)

    @classmethod
    def protocol_descr(cls):
        """Return a dataframe with descriptions
        and call signature of all registered protocols
        """
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
def xenon_resource(name: str,
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
    if isinstance(version, str) and version.startswith('global'):
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
