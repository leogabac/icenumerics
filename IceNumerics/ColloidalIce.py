import math
import numpy as np
import copy
from matplotlib import pyplot as plt
from matplotlib import patches
import scipy.spatial as spa

from IceNumerics.Vector import Vector
        
from IceNumerics.SquareSpinIceGeometry import *

from IceNumerics.Parameters import *

from IceNumerics.Spins import *

class ColloidInTrap():
    
    ## a Spin is defined by three vectors in R3 space.
    # the vector center gives the position of the center of the trap
    # the vector direction goes from one stable position to another.
    # the vector colloid goes from the center of the trap to the position of the
    # colloid
    # vectors in R3 space are defined as tuples
    
    def __init__(self,
                 center = Vector((0,0,0)),
                 direction = Vector((1,0,0)),
                 colloid = Vector((0,0,0)),
                 **kargs):

        self.colloidparams = ColloidParameters(**kargs)                
        if 'ColloidParameters' in kargs:
            self.colloidparams = kargs['ColloidParameters']
        
        self.geometry = TrapGeometry(**kargs);
        if 'Geometry' in kargs: self.geometry = kargs['Geometry']

        if 'Ordering' in kargs: self.ordering = kargs['Ordering']

        if center.__class__.__name__=='Vector':
            self.center = center
        elif center.__class__.__name__=='tuple':
            self.center = Vector((center))
        else:
            error("Invalid input class for object ColloidInTrap")
            
        if direction.__class__.__name__=='Vector':
            self.direction = direction
        elif center.__class__.__name__=='tuple':
            self.direction = Vector((direction))
        else:
            error("Invalid input class for object ColloidInTrap")

        if colloid.__class__.__name__=='Vector':
            self.colloid = colloid
        elif center.__class__.__name__=='tuple':
            self.colloid = Vector((colloid))
        else:
            error("Invalid input class for object ColloidInTrap")

        
    def __str__(self):
        return("Colloid is in [%d %d %d], trap is [%d %d %d %d %d %d]\n" %\
               (tuple(self.colloid)+tuple(self.center)+tuple(self.direction)))
               
    def display(self, ax1):
        
        X=self.center[0]
        Y=self.center[1]
        # D is the vector that goes fom the center to the positive spin
        # I am not sure what positive spin means in this context
        DX=self.direction[0]/2
        DY=self.direction[1]/2
        # P is the vector that goes from the center to the colloid
        PX=self.colloid[0]
        PY=self.colloid[1]
        
        #Discriminant = self.colloid.dot(self.direction)
        #Discriminant = Discriminant/abs(Discriminant)
        
        #DX = DX*Discriminant
        #DY = DY*Discriminant
        
        # W = math.sqrt(DX**2+DY**2)+2*self.geometry.height
        W = self.geometry.stiffness*5e7
        
        ax1.plot(X,Y,'k')
        ax1.add_patch(patches.Circle(
            (X-DX,Y-DY),radius = W,
            ec='g', fc='none'))
        ax1.add_patch(patches.Circle(
            (X+DX,Y+DY),radius = W,
            ec='y', fc='none'))
        ax1.add_patch(patches.Circle(
            (X+PX,Y+PY), radius = W/3, ec='k', fc = 'none'))
        # print([DX,DY])
        
    def flip(self):
        """flips the ColloidInTrap by inverting its direction and its colloid attributes. Returns fliped object"""
        cp = copy.deepcopy(self);
        
        cp.direction = self.direction*(-1)
        cp.colloid = self.colloid*(-1)
        return cp
        
    def bias(self, vector):
        """ 
        Flips the ColloidInTrap to make it point in the direction of vector (dot(colloid,vector)>0). Returns fliped object        
        """
        if not (vector.__class__.__name__=="Vector" \
            or vector.__class__.__name__=="ndarray" \
            or vector.__class__.__name__=="list" \
            or vector.__class__.__name__=="tuple"):
                raise ValueError("The vector argument must be either a Vector, ndarray, list or tuple")
        # if vector is 2D, append zero
        if len(vector)==2:
            if vector.__class__.__name__=="Vector":
                vector = Vector(tuple(vector)+Vector([0]))
            if vector.__class__.__name__=="tuple":
                vector = vector+0
            if vector.__class__.__name__=="ndarray":
                vector = np.hstack((vector,0))
            if vector.__class__.__name__=="list":
                vector = vector+[0]
        elif len(vector)>3:
            raise ValueError("The vector argument has to be 2 or 3 dimentions")
        
        cp = copy.deepcopy(self);

        # Flip if when direction is opposite to bias
        if vector.__class__.__name__=="Vector" or vector.__class__.__name__=="ndarray":
            if vector.dot(self.direction)<0:
                cp = self.flip()
        elif vector.__class__.__name__=="list" or vector.__class__.__name__=="tuple":
            if Vector(vector).dot(self.direction)<0:
                cp = self.flip()
        return cp
        
class ColloidalIce(dict):
    # Physical Attributes
    # Geometry

    def __init__(self,*args,**kargs):

        self.worldparams = WorldParameters(**kargs)
        geometry = TrapGeometry(**kargs)
        
        self.lattice = args[0].lattice
        ordering = args[0].ordering
        
        for s in args[0]:
            # Unlike Spins, you can have several traps in the same place.
            # Each trap has its own colloid. This is a (inefficient) way of having
            # several colloids in one trap.
            
            c = ColloidInTrap(
                args[0][s].center,
                args[0][s].direction*geometry.trap_sep_ratio,
                args[0][s].direction*(geometry.trap_sep_ratio/2),
                Ordering = ordering,**kargs)
            
            self[Vector(args[0][s].center)] = c

        self.worldparams.set_region(self)


        
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
         
    def FrameDataToColloids(self,FrameData,Run):
        
        NumberOfRuns = np.max(FrameData['type']);
        
        if Run>=NumberOfRuns:
            raise ValueError("You are asking for a run that doesn't exist")
            
        FrameDataSort = FrameData[FrameData['id'].argsort()]

        X = FrameData['x'][FrameData['type']==Run+1]
        Y = FrameData['y'][FrameData['type']==Run+1]

        Xc = FrameData['x'][FrameData['type']==NumberOfRuns]
        Yc = FrameData['y'][FrameData['type']==NumberOfRuns]
        
        Centers = np.array([np.array(self[c].center) for c in self])

        for (xc,yc,x,y) in zip(Xc,Yc,X,Y):
            v = Vector((xc*1000,yc*1000,0))
            key = Vector(Centers[np.argmin(np.sqrt(np.sum((Centers-v)**2,1)))])
            self[key].colloid = Vector(((x-xc)*1000,(y-yc)*1000,0))
            
            dot = self[key].colloid.dot(self[key].direction);
            if np.sign(dot) != 0:
                self[key].direction = Vector((np.sign(dot)*self[key].direction))
        return self
    
    def CalculateEnergy(self):
        """ Calculates the sum of the inverse cube of all the inter particle distances.
        For this it uses the spatial package of scipy which shifts this calculation to a compiled program and it's therefore faster.
        """
        colloids = np.array([np.array(self[c].center+self[c].colloid) for c in self])
        self.energy = sum(spa.distance.pdist(colloids)**(-3))
        return self.energy
