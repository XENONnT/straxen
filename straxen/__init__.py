# mypy: disable-error-code="no-redef"
__version__ = "3.2.7"

import numpy as np
import pandas as pd
import strax
import strax.utils as strax_utils

# NumPy >= 2.4 removed np.in1d; keep an alias for dependencies (e.g. strax)
# that still call it.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# pandas >= 3 may return ExtensionArray (e.g. ArrowStringArray) for `.values`.
# Make strax.to_str_tuple accept those arrays.
_orig_to_str_tuple = strax_utils.to_str_tuple


def _to_str_tuple_compat(x):
    if isinstance(x, pd.api.extensions.ExtensionArray):
        return tuple(x.tolist())
    return _orig_to_str_tuple(x)


strax_utils.to_str_tuple = _to_str_tuple_compat
strax.to_str_tuple = _to_str_tuple_compat

from utilix import uconfig
from .common import *

from .itp_map import *
from .matplotlib_utils import *
from .mini_analysis import *
from .misc import *

from .scada import *
from .bokeh_utils import *
from .config.url_config import *

from . import plugins
from .plugins import *

from . import storage
from .storage import *

from . import analyses

from . import config

from . import units

# Do not make all contexts directly available under straxen.
# Otherwise, we have straxen.demo() etc.
from . import contexts

from . import test_utils
from .test_utils import *

from . import docs_utils
from .docs_utils import *

from . import daq_core

try:
    from . import holoviews_utils
    from .holoviews_utils import *
except ModuleNotFoundError:
    pass

from .entry_points import load_entry_points

load_entry_points()
del load_entry_points
