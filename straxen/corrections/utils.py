import re
import inspect
from urllib.parse import urlparse, parse_qs
from ast import literal_eval


class TypeDispatch:
    """Simple single dispatch.
       Implementation is copied from dask source code
    """

    def __init__(self, name=None):
        self._lookup = {}
        self._lazy = {}
        if name:
            self.__name__ = name

    def register(self, type, func=None):
        """Register dispatch of `func` on arguments of type `type`"""

        def wrapper(func):
            if isinstance(type, tuple):
                for t in type:
                    self.register(t, func)
            else:
                self._lookup[type] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def register_lazy(self, toplevel, func=None):
        """
        Register a registration function which will be called if the
        *toplevel* module (e.g. 'pandas') is ever loaded.
        """

        def wrapper(func):
            self._lazy[toplevel] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def dispatch(self, cls):
        """Return the function implementation for the given ``cls``"""
        # Fast path with direct lookup on cls
        lk = self._lookup
        try:
            impl = lk[cls]
        except KeyError:
            pass
        else:
            return impl
        # Is a lazy registration function present?
        toplevel, _, _ = cls.__module__.partition(".")
        try:
            register = self._lazy.pop(toplevel)
        except KeyError:
            pass
        else:
            register()
            return self.dispatch(cls)  # recurse
        # Walk the MRO and cache the lookup result
        for cls2 in inspect.getmro(cls)[1:]:
            if cls2 in lk:
                lk[cls] = lk[cls2]
                return lk[cls2]
        raise TypeError("No dispatch for {0}".format(cls))

    def __call__(self, arg, *args, **kwargs):
        """
        Call the corresponding method based on type of argument.
        """
        meth = self.dispatch(type(arg))
        return meth(arg, *args, **kwargs)

    @property
    def __doc__(self):
        try:
            func = self.dispatch(object)
            return func.__doc__
        except TypeError:
            return "Single Dispatch for %s" % self.__name__


