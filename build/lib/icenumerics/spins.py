import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
        
from icenumerics.geometry import *
from icenumerics import ureg

class spin():
    """ 
    A spin is defined by two vectors in R3 space.
    The vector center gives the position of the center of the spin
    The vector direction gives the dipole moment of the spin.
    """
    
    def __init__(self,center,direction):
        
        self.center = np.array(center)*center.units
        self.direction = np.array(direction)*center.units
        
    def __str__(self):
        return("Spin with Center at [%d %d %d] and Direction [%d %d %d]\n" %\
               (tuple(self.center)+tuple(self.direction)))

    def display(self,ax1):
        
        X=self.center[0].magnitude
        Y=self.center[1].magnitude
        DX=self.direction[0].magnitude*0.3
        DY=self.direction[1].magnitude*0.3
        W = np.sqrt(DX**2+DY**2)
        ax1.plot([X],[Y],'b')
        ax1.add_patch(patches.Arrow(X-DX,Y-DY,2*DX,2*DY,width=W,fc='b'))

class spins(list): 
    """ `spins` is a very general class that contains a list of spin objects. The only feature of this list is that it is created from the centers and directions of the spins, and also that it contains a ´display´ method. """ 
    
    def __init__(self, centers = [], directions = None):
        """To initialize, we can give the centers and directions of the spins contained. However, we can also initialize an empty list, and then populate it using the `extend` method """
        
        if len(centers)>0:
            self = self.extend([spin(c,d) for (c,d) in zip(centers,directions)])
        
        
    def display(self,ax = None, ix = False):
        """ This displays the spins in a pyplot axis. The ix parameter allows us to obtain the spins index, which is useful to access specific indices."""
        if not ax:
            fig1, ax = plt.subplots(1,1)            

        for s in self:
            s.display(ax)

        plt.axis("equal")

    def create_lattice(self, geometry, size,
        lattice_constant = 1, border = "closed spin"):
        """ Creates a lattice of spins. 
        The geometry can be:
            * "square"
            * "honeycomb"
        The border can be 
            * "closed spin":
            * "closed vertex" 
            * "periodic" 
        """
        
        latticeunits = lattice_constant.units

        if geometry == "square":
            center, direction = square_spin_ice_geometry(
                size[0], size[1], lattice_constant.magnitude,
                "Random", Ratio = 1, Boundary = border
            )
                    
        elif geometry == "honeycomb":
            center, direction = honeycomb_spin_ice_geometry(
                size[0], size[1], lattice_constant.magnitude,
                "Random"
            )
            
        else: 
            raise(ValueError(geometry+" is not a supporteed geometry."))
        
        self.__init__(center*latticeunits,direction*latticeunits)
        self.lattice = lattice_constant
        self.ordering = "random"

