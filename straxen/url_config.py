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

def parse_val(val):
    try:
        val = literal_eval(val)
    except ValueError:
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
            self.type = OMITTED # do not enforce type on the URL
        if cache:
            cache_len = 1_000_000 if cache is True else int(cache) 
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
        protocol, _, path =  url.partition(self.SCHEME_SEP)

        # find the corresponding protocol method
        meth = self._LOOKUP.get(protocol, None)
        if meth is None:
            # unrecongnized protocol
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

    def split_url_kwargs(self, url):
        """split a url into path and kwargs
        """
        path, _, _ = url.partition(self.QUERY_SEP)
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
                kwargs[k] = map(parse_val, v)
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

        kwargs = {}
        for k, v in url_kwargs.items():
            if isinstance(v, str) and v.startswith(self.PLUGIN_ATTR_PREFIX):
                # kwarg is referring to a plugin attribute, lets fetch it
                kwargs[k] = getattr(plugin, v[len(self.PLUGIN_ATTR_PREFIX):], v)
            else:
                # kwarg is a literal, add its value to the kwargs dict
                kwargs[k] = v

        # construct a deterministic hash key
        key = strax.deterministic_hash((url, kwargs))

        # fetch from cache if exists
        value = self.cache.get(key, None)

        # not in cache, letch fetch it
        if value is None:
            value = self.dispatch(url, **kwargs)
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


@URLConfig.register('cmt')
def get_correction(name: str, run_id: str=None, version: str='ONLINE', detector: str='nt', **kwargs):
    '''Get value for name from CMT
    '''
    if run_id is None:
        raise ValueError('Attempting to fetch a correction without a run id.')
    return straxen.get_correction_from_cmt(run_id, (name, version, detector=='nt'))


@URLConfig.register('resource')
def get_resource(name: str, fmt: str='text', **kwargs):
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
    return container[take]


@URLConfig.register('format')
def format_arg(arg: str, **kwargs):
    '''apply pythons builtin format function to a string
    '''
    return arg.format(**kwargs)
