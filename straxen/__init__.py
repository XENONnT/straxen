__version__ = '0.12.3'

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
from .itp_map import *
from .rundb import *
from .online_monitor import *
from .matplotlib_utils import *
from .mini_analysis import *
from .misc import *
from .scada import *

from .get_corrections import *
from .hitfinder_thresholds import *
from .corrections_services import *

from . import plugins
from .plugins import *

from . import analyses

# Do not make all contexts directly available under straxen.
# Otherwise we have straxen.demo() etc.
from . import contexts
