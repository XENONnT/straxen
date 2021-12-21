import sys
import json
from typing import Container
import strax
import fsspec
import pandas as pd
import straxen
import inspect
from urllib.parse import urlparse, parse_qs
from ast import literal_eval
from functools import lru_cache
from strax.config import OMITTED

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
        kwargs = self.filter_kwargs(meth, kwargs)

        return meth(arg, *args, **kwargs)

    def preprocessor_dispatch(self, url, **kwargs):
        """

        """
        if self.SCHEME_SEP not in url:
            return url, {}

        # separate the protocol name from the path
        protocol, _, path = url.partition(self.SCHEME_SEP)

        # find the corresponding protocol method
        meth = self._PREPROCESSORS.get(protocol, None)

        # if meth is None:
        # no registered preprocessor
        # skip preprocessing for this protocol
        # and preprocess the next
        path, extra_kwargs = self.preprocessor_dispatch(path, **kwargs)
        kwargs.update(extra_kwargs)

        if protocol:
            url = f"{protocol}{self.SCHEME_SEP}{path}"

        if meth is None:
            return url, extra_kwargs

        if self.SCHEME_SEP in path:
            # url contains a nested protocol
            # first call sub-protocol

            # FIXME: dispatch may depend on runtime evaluated
            # kwargs, maybe check if any kwargs are urls,
            # meaning they will only be evaluated at runtime
            arg = self.dispatch(path, **kwargs)

        else:
            # we are at the end of the chain
            # method should be called with path as argument
            arg = path

        # filter kwargs to pass only the kwargs
        #  accepted by the method.
        kwargs = self.filter_kwargs(meth, kwargs)

        # run the protocol
        url = meth(url, arg, **kwargs)

        if not isinstance(url, str):
            raise PreProcessorError('Preprocessor for {protocol} protocol'
                                ' has returned incompatible value.'
                                ' Got {processed}, should be a url string'
                                ' or a tuple(str,dict)')
        
        url, url_kwargs = self.split_url_kwargs(url)

        extra_kwargs.update(url_kwargs)

        return url, extra_kwargs

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
        if isinstance(value, str) and value.startswith(self.PLUGIN_ATTR_PREFIX):
            # kwarg is referring to a plugin attribute, lets fetch it
            attr = value[len(self.PLUGIN_ATTR_PREFIX):]
            default = config.get(attr, value)
            return getattr(plugin, attr, default)

        if isinstance(value, list):
            return [self.fetch_attribute(plugin, v, **config) for v in value]

        # kwarg is a literal, add its value to the kwargs dict
        return value

    @classmethod
    def format_url_kwargs(cls, url, **kwargs):
        url, extra_kwargs = cls.split_url_kwargs(url)
        kwargs.update(extra_kwargs)
        arg_str = "&".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return url+cls.QUERY_SEP+arg_str

    @classmethod
    def build_url(cls, *args, **kwargs):
        if not args:
            return cls.format_url_kwargs('', **kwargs)
        if len(args) == 1:
            return cls.format_url_kwargs(args[0], **kwargs)

        protocol_prefix = args[0] + cls.SCHEME_SEP
        url = cls.build_url(*args[1:], **kwargs)
        return protocol_prefix + url

    @classmethod
    def parse_url(cls, url):
        if not isinstance(url, str):
            # if the value is not a string it is evaluated
            # as a literal config and returned as is.
            return url

        if cls.SCHEME_SEP not in url:
            # no protocol in the url so its evaluated
            # as string-literal config and returned as is
            return url

        # separate the protocol name from the path
        protocol, _, path = url.partition(cls.SCHEME_SEP)

        # find the corresponding protocol method
        meth = cls._LOOKUP.get(protocol, None)
        if meth is None:
            # unrecognized protocol
            # evaluate as string-literal
            return url

        
        arg, kwargs = cls.split_url_kwargs(path)

        if cls.SCHEME_SEP in path:
            # url contains a nested protocol
            # first parsce sub-protocol
            arg = cls.parse_url(path)

        # filter kwargs to pass only the kwargs
        #  accepted by the method.
        
        kwargs = cls.filter_kwargs(meth, kwargs)

        return {
            'protocol': protocol,
            'argument': arg,
            'kwargs': kwargs,
            }

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
        url, url_kwargs = self.split_url_kwargs(url)

        kwargs = {k: self.fetch_attribute(plugin, v)
                  for k, v in url_kwargs.items()}

        # construct a deterministic hash key
        key = strax.deterministic_hash((url, kwargs))

        # fetch from cache if exists
        value = self.cache.get(key, None)

        # not in cache, lets fetch it
        if value is None:
            value = self.dispatch(url, **kwargs)
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

        if self.SCHEME_SEP not in url:
            # no protocol in the url so its evaluated
            # as string-literal config and returned as is
            return url

        # separate out the query part of the URL which
        # will become the method kwargs
        url, url_kwargs = self.split_url_kwargs(url)

        # fetch any kwargs that reference other configs
        kwargs = {}
        for k, v in url_kwargs.items():
            kwargs[k] = self.fetch_attribute(self._PLUGIN_CLASS, v, **config)
        
        # dispatch any protocol preprocessors
        url, extra_kwargs = self.preprocessor_dispatch(url, **kwargs)

        # add/update any extra kwargs the preprocessors produced 
        url_kwargs.update(extra_kwargs)

        # build the modified URL from the preprocessor results
        url = self.format_url_kwargs(url, **url_kwargs)

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
def get_correction(name: str, run_id: str = None, version: str = 'ONLINE', detector: str = 'nt'):
    '''Get value for name from CMT
    '''
    if run_id is None:
        raise ValueError('Attempting to fetch a correction without a run id.')
    return straxen.get_correction_from_cmt(run_id, (name, version, detector == 'nt'))


@URLConfig.register('resource')
def get_resource(name: str, fmt: str = 'text'):
    '''Fetch a straxen resource
    '''
    return straxen.get_resource(name, fmt=fmt)


@URLConfig.register('fsspec')
def read_file(path: str, **kwargs):
    '''Support fetching files from arbitrary filesystems
    '''
    with fsspec.open(path, **kwargs) as f:
        content = f.read()
    return content


@URLConfig.register('json')
def read_json(content: str, **kwargs):
    ''' Load json string as a python object
    '''
    return json.loads(content)


@URLConfig.register('take')
def get_key(container: Container, take=None, **kwargs):
    ''' return a single element of a container
    '''
    if take is None:
        return container
        
    if not isinstance(take, list):
        take = [take]

    for t in take:
        container = container[t]

    return container


@URLConfig.register('format')
def format_arg(arg: str, **kwargs):
    ''' apply pythons builtin format function to a string
    '''
    return arg.format(**kwargs)


@URLConfig.register_preprocessor('cmt')
def replace_global_version(url, correction, version=''):
    if version.startswith('global'):
        local_versions = get_cmt_local_versions(version)
        v = local_versions.get(correction, version)
        url += f'?version={v}'
    return url


@lru_cache(maxsize=2)
def get_cmt_local_versions(global_version):
    cmt = straxen.CorrectionsManagementServices()
    return cmt.get_local_versions(global_version)

