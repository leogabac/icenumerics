import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

from IceNumerics.Vector import Vector
        
from IceNumerics.SquareSpinIceGeometry import *
from IceNumerics.HoneycombSpinIceGeometry import *
from IceNumerics.Parameters import TrapGeometry
from IceNumerics.Parameters import WorldParameters
from IceNumerics.Parameters import SimulationParameters



class Spin():
    
    ## a Spin is defined by two vectors in R3 space.
    # the vector center gives the position of the center of the spin
    # the vector direction gives the dipole moment of the spin.
    # vectors in R3 space are defined as tuples
    
    def __init__(self,center=Vector((0,0,0)),direction=Vector((0,0,0))):
        
        if center.__class__.__name__=='Vector':
            self.center = center
        elif center.__class__.__name__=='tuple':
            self.center = Vector((center))
        else:
            error("Invalid input class for object spin")
            
        if direction.__class__.__name__=='Vector':
            self.direction = direction
        elif center.__class__.__name__=='tuple':
            self.direction = Vector((direction))
        else:
            error("Invalid input class for object spin")
        
    def __str__(self):
        return("Spin with Center at [%d %d %d] and Direction [%d %d %d]\n" %\
               (tuple(self.center)+tuple(self.direction)))

    def display(self,ax1):
        X=self.center[0]
        Y=self.center[1]
        DX=self.direction[0]*0.3
        DY=self.direction[1]*0.3
        W = math.sqrt(DX**2+DY**2)
        ax1.plot([X],[Y],'b')
        ax1.add_patch(patches.Arrow(X-DX,Y-DY,2*DX,2*DY,width=W,fc='b'))

class Spins(dict):  
    
    def __init__(self,*args):
        for s in args:
            if s.center in self:
                # if this spin is alread y in the system,
                # then average both directions
                self[s.center].direction+=s.direction
                self[s.center].direction*=0.5
            else:
                self[s.center] = s

    def __str__(self):
        
        PrntStr = ""
        for s in self:
            PrntStr += \
                self[s].__str__()
                
        return(PrntStr)
        
    def display(self,DspObj = False):

        d=0.1
        AxesLocation = [d,d,1-2*d,1-2*d]
                
        if not DspObj:
            fig1 = plt.figure(1)
            ax1 = plt.axes(AxesLocation)
        elif DspObj.__class__.__name__ == "Figure":
            fig1 = DspObj
            ax1 = plt.axes(AxesLocation, frameon=0)
            fig1.patch.set_visible(False)
        elif DspObj.__class__.__name__ == "Axes":
            ax1 = DspObj
            fig1.patch.set_visible(False)
        elif DspObj.__class__.__name__ == "AxesSubplot":
            ax1 = DspObj

        for s in self:
            self[s].display(ax1)

        plt.axis("equal")

        if DspObj.__class__.__name__ == "Figure":
            ax1 = DspObj
            fig1.patch.set_visible(False) 
            plt.show(block = True)
        
class SquareSpinIce(Spins):
    
    def __init__(self,Sx=1,Sy=1,**kargs):
        """ SquareSpinIce Takes the following arguments upon definition:
        Sx (1),
        Sy (1),
        --------------
        Keyword Args:
        Periodic (False)
        Lattice (1)
        Ordering [GroundState,Bias,(Random)]
        Boundary [(ClosedSpin),ClosedVertex]
        """

        # Parse Keyword Arguments
        
        if 'Lattice' in kargs: self.lattice = kargs['Lattice']
        else: self.lattice = 1;

        if 'Ordering' in kargs: self.ordering = kargs['Ordering']
        else: self.ordering = 'Random'

        if 'Ratio' in kargs: self.Ratio = kargs['Ratio']
        else: self.Ratio = 1
        
        if 'Boundary' in kargs: self.Boundary = kargs['Boundary']
        else: self.Boundary = "ClosedSpin"

        Center,Direction = SquareSpinIceCalculateGeometry(
            Sx,Sy,self.lattice,self.ordering,self.Ratio,self.Boundary)

        Direction = Direction*self.lattice
        
        S = tuple(Spin(tuple(c),tuple(d)) for c, d in zip(Center,Direction))
        
        super(SquareSpinIce,self).__init__(*S)

class HoneycombSpinIce(Spins):
    
    def __init__(self,Sx=1,Sy=1,**kargs):
        # HoneycombSpinIce Takes the following arguments upon definition:
        # Sx (1),
        # Sy (1),
        # --------------
        # Keyword Args:
        # Periodic (False)
        # Lattice (1)
        # Ordering [GroundState,Bias,(Random)]

        # Parse Keyword Arguments
        
        if 'Lattice' in kargs: self.lattice = kargs['Lattice']
        else: self.lattice = 1;

        if 'Ordering' in kargs: self.ordering = kargs['Ordering']
        else: self.ordering = 'Random'

        Center,Direction = HoneycombSpinIceCalculateGeometry(
            Sx,Sy,self.lattice,self.ordering)

        Direction = Direction*self.lattice
        
        S = tuple(Spin(tuple(c),tuple(d)) for c, d in zip(Center,Direction))
                
        super(HoneycombSpinIce,self).__init__(*S)

