import collections

import strax
export, __all__ = strax.exporter()


@export
def flatten_dict(d, separator=':', _parent_key=''):
    """Flatten nested dictionaries into a single dictionary,
    indicating levels by separator.
    Don't set _parent_key argument, this is used for recursive calls.
    Stolen from http://stackoverflow.com/questions/6027558
    """
    items = []
    for k, v in d.items():
        new_key = _parent_key + separator + k if _parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v,
                                      separator=separator,
                                      _parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
