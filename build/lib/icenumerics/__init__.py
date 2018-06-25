from pint import UnitRegistry

import sys

try: 
    from .magcolloids import magcolloid as mc
except ImportError as e:
    try: 
        import magcolloid as mc
    except ImportError as e:
        raise ImportError

ureg = mc.ureg

from icenumerics.spins import *
from icenumerics.colloidalice import *
from icenumerics.vertices import *