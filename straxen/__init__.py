__version__ = '0.14.2'

from warnings import warn
# load configuration file using utilix
try:
    from utilix import uconfig
# if no utilix config, get a RuntimeError
# we don't want this to break straxen, but it does print a warning statement
except (FileNotFoundError, RuntimeError) as e:
    warn('Warning, utilix config file not loaded properly. copy '
         '/project2/lgrandi/xenonnt/.xenon_config to your HOME directory!',
         UserWarning)
    uconfig = None

from .common import *
# contexts.py below
from .corrections_services import *
from .get_corrections import *
from .hitfinder_thresholds import *
from .itp_map import *
from .matplotlib_utils import *
from .mini_analysis import *
from .misc import *
from .mongo_storage import *
from .online_monitor import *
from .rundb import *
from .scada import *
from .bokeh_utils import *


from . import plugins
from .plugins import *

from . import analyses

# Do not make all contexts directly available under straxen.
# Otherwise we have straxen.demo() etc.
from . import contexts
