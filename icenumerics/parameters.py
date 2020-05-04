import numpy as np
import warnings
from . import ureg

class trap():
    """ The trap geometry object contains:
    TrapSep: Distance between stable points of the traps [in nm]
    Height: Hill height of the trap [in nm]
    Stiffness: Stiffness of the trap [in pNnm].
    """
    
    # Trap Separation could be determined also by the spin dipole.
    # I need to think how to put them together. 

    def __init__(self,
        trap_sep = 10e3*ureg.nm,
        height = 200*ureg.nm,
        stiffness = 1.2e-4 * ureg.pN/ureg.nm,
        cutoff = np.inf*ureg.um):
        """ This initializes a trap parameter set. """
        
        self.trap_sep = trap_sep;
        self.height = height;
        self.stiffness = stiffness;
        self.cutoff = cutoff;

class particle():
    """
    The diffusion coefficient can be given instead o the drag.
    In that case, the temperature is also needed to calculatelm the drag. 
    This represents the density mismatch of the particle in the solvent
    """
    def __init__(self, 
                radius = 2*ureg.um,
                susceptibility = 1,
                drag = 4e6*ureg.pN/(ureg.um/ureg.s), 
                diffusion = None, temperature = None, 
                density = 1000*ureg.kg/ureg.m**3):
        """Initializes a particle type. """
        
        if diffusion:
            
            kB = (1.38064852e-23*ureg.J/ureg.K).to(ureg.pN*ureg.nm/ureg.K)
            
            KbT = kB*temperature
            drag = KbT/diffusion
            
        self.radius = radius.to(ureg.um)
        self.susceptibility = susceptibility
        self.drag = drag.to(ureg.pg/ureg.us)
        self.mass = (density*4/3*np.pi*(radius)**3).to(ureg.pg)
        
        damp = 1e-3*ureg.us
        
        self.drag_mass = (drag*damp).to(ureg.pg)
        
class world():
    
    def __init__(self,
            temperature = 300*ureg.K,
            field = 20*ureg.mT,
            dipole_cutoff = 200*ureg.um,
            enforce2d = True,
            boundaries = ["s","s","p"]):
        # The force parameter adds a biasing force to the simulation.
        
        self.temperature = temperature
        self.kB = (1.38064852e-23*ureg.J/ureg.K).to(ureg.pN*ureg.nm/ureg.K)
        self.enforce2d = enforce2d
        self.field = field
        self.dipole_cutoff = dipole_cutoff
        self.boundaries = boundaries
