from pint import UnitRegistry

import sys

try: 
    from .magcolloids import magcolloids as mc
except ImportError as e:
    try: 
        import magcolloids as mc
    except ImportError as e:
        raise ImportError

ureg = mc.ureg

from icenumerics.spins import *
from icenumerics.colloidalice import *
from icenumerics.vertices import *
from icenumerics.trajectory import *

__version__ = "0.1.9"