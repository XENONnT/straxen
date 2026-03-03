import numpy as np
import pandas as pd
import strax
import strax.utils as strax_utils


def apply_runtime_compat():
    """Apply runtime compatibility patches for dependency API drift."""
    # NumPy >= 2.4 removed np.in1d; keep an alias for dependencies
    # (e.g. strax) that still call it.
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    # pandas >= 3 may return ExtensionArray (e.g. ArrowStringArray) for
    # `.values`, while strax.to_str_tuple expects ndarray/Series/tuple/list.
    current = strax_utils.to_str_tuple
    if getattr(current, "_straxen_patched", False):
        return

    def _to_str_tuple_compat(x):
        if isinstance(x, pd.api.extensions.ExtensionArray):
            return tuple(x.tolist())
        return current(x)

    _to_str_tuple_compat._straxen_patched = True
    strax_utils.to_str_tuple = _to_str_tuple_compat
    strax.to_str_tuple = _to_str_tuple_compat
