__version__ = '0.11.0'

# load configuration file using utilix
try:
    from utilix.config import Config
    uconfig = Config()
except FileNotFoundError:
    print("Warning: no xenon configuration file found")


from .common import *
from .itp_map import *
from .rundb import *
from .online_monitor import *
from .matplotlib_utils import *
from .mini_analysis import *
from .misc import *

from .get_corrections import *
from .hitfinder_thresholds import *
from .corrections_services import *

from . import plugins
from .plugins import *

from . import analyses

# Do not make all contexts directly available under straxen.
# Otherwise we have straxen.demo() etc.
from . import contexts



