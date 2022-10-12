__version__ = '1.8.3'

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
# Otherwise we have straxen.demo() etc.
from . import contexts

from . import test_utils
from .test_utils import *

try:
    from . import holoviews_utils
    from .holoviews_utils import *
except ModuleNotFoundError:
    pass
