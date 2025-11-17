# skewerjax/_version.py
__version__ = "0.0.1"

# Eagerly expose lightweight namespaces
from . import config as config
from . import constants as constants
from . import optical_depth as optical_depth
from . import utils as utils

__all__ = ["__version__", "config", "constants", "optical_depth", "utils"]
