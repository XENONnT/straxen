__version__ = '2.0.7'

from utilix import uconfig
from .common import *
# contexts.py below
from .corrections_services import *
from .get_corrections import *

from .itp_map import *
from .matplotlib_utils import *
from .mini_analysis import *
from .misc import *

from .scada import *
from .bokeh_utils import *
from .url_config import *

from . import legacy
from .legacy import *

from . import plugins
from .plugins import *

from . import storage
from .storage import *

from . import analyses

# Do not make all contexts directly available under straxen.
# Otherwise, we have straxen.demo() etc.
from . import contexts

from . import test_utils
from .test_utils import *

from . import daq_core

try:
    from . import holoviews_utils
    from .holoviews_utils import *
except ModuleNotFoundError:
    pass

from .entry_points import load_entry_points

import warnings

warnings.warn("From version 2.0.8 forward in straxen the URLs will be sorted alphabetically before hashing, If you wish to access data the was processed using pervious versions of straxen, its recomended you use a previous version, otherwise you will have to reprocess your data.", FutureWarning)

load_entry_points()
del load_entry_points
