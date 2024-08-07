import re


_pattern_type = type(re.compile(""))


def normalize(r):
    """Normalize a regular expression by ensuring that it is wrapped with:
    '^' and '$'

    Parameters
    ----------
    r : str or Pattern
        The pattern to normalize.

    Returns
    -------
    p : Pattern
        The compiled regex.
    """
    if isinstance(r, _pattern_type):
        r = r.pattern
    return re.compile(r.lstrip("^").rstrip("$"))


class RegexDispatcher(object):
    """Regular Expression Dispatcher.

    >>> f = RegexDispatcher('f')

    >>> f.register('\d*')
    ... def parse_int(s):
    ...     return int(s)

    >>> f.register('\d*\.\d*')
    ... def parse_float(s):
    ...     return float(s)

    Set priorities to break ties between multiple matches.
    Default priority is set to 10

    >>> f.register('\w*', priority=9)
    ... def parse_str(s):
    ...     return s

    >>> type(f('123'))
    int

    >>> type(f('123.456'))
    float

    """

    def __init__(self, name):
        self.name = name
        self.funcs = {}

    def add(self, regex, func):
        # self.funcs[normalize(regex)] = func
        norm_regex = normalize(regex)
        if norm_regex not in self.funcs:
            self.funcs[norm_regex] = []
        self.funcs[norm_regex].append(func)

    def register(self, regex):
        """Register a new handler in this regex dispatcher.

        Parameters
        ----------
        regex : str or Pattern
            The pattern to match against.
        priority : int, optional
            The priority for this pattern. This is used to resolve ambigious
            matches. The highest priority match wins.

        Returns
        -------
        decorator : callable
            A decorator that registers the function with this RegexDispatcher
            but otherwise returns the function unchanged.

        """

        def _(func):
            self.add(regex, func)
            return func

        return _

    def dispatch(self, s):
        # funcs = [func for r, func in self.funcs.items() if r.match(s)]
        # return funcs
        funcs = [f for r, funcs in self.funcs.items() if r.match(s) for f in funcs]
        return funcs

    def __call__(self, s, *args, **kwargs):
        if isinstance(s, str):
            funcs = self.dispatch(s)
            for func in funcs:
                func(s, *args, **kwargs)

    @property
    def __doc__(self):
        # take the min to give the docstring of the last fallback function
        return min(self.priorities.items(), key=lambda x: x[1])[0].__doc__
