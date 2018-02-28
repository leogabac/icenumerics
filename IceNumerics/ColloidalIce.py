import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import scipy.spatial as spa

from IceNumerics.SquareSpinIceGeometry import *
from IceNumerics.Parameters import *
from IceNumerics.Spins import *

class colloid_in_trap():
    """
    An object ColloidInTrap represents a colloidal particle in a bistable trap. 
    It has three main properties:
    * center is the center of the bistable trap. 
    * direction is a vector (whose magnitude is not important) that points from one stable
        position to the other.
    * colloid is a vector that indicates where the colloid is placed with respect to the center.
    
    Each of these quantities are represented by lists of three elements, which correspond to 
    vectors in R3 space. 
    
    A colloid_in_trap object also has the properties:
    * colloid_properties:
        * susceptibility
        * diffusion
        * diameter
        * rel_density
        * volume
    * trap_properties
        * trap_sep: The distance between the traps.
        * height
        * stiffness
        * spread
    """
    def __init__(self, 
        center = [0,0,0], 
        direction = [1,0,0], 
        **kargs):
        """ initializes a colloid_in_trap object """
        
        self.colloidparams = ColloidParameters(**kargs)                
        if 'ColloidParameters' in kargs:
            self.colloidparams = kargs['ColloidParameters']
        
        self.geometry = TrapGeometry(**kargs);
        if 'Geometry' in kargs: self.geometry = kargs['Geometry']

        if 'Ordering' in kargs: self.ordering = kargs['Ordering']

        if center.__class__.__name__=='Vector':
            self.center = np.array(center)
   
        if direction.__class__.__name__=='Vector':
            self.direction = np.array(direction)
        
        """ Make direction unitary"""
        self.direction = np.array(self.direction)
        self.direction = self.direction/np.sqrt(sum(self.direction**2))
        
        """ Colloid position is given by the direction vectors"""
        self.colloid = self.direction * self.geometry.trap_sep/2
        
    def __str__(self):
        """ Prints a string which represents the colloid_in_trap """
        return("Colloid is in [%d %d %d], trap is [%d %d %d %d %d %d]\n" %\
               (tuple(self.colloid)+tuple(self.center)+tuple(self.direction)))
               
    def display(self, ax1=False):
        """ Draws a figure with the trap and the colloid inside it"""
        if not ax1:
            fig, ax1 = plt.subplots(1,1)
            
        X=self.center[0]
        Y=self.center[1]
        # D is the vector that goes fom the center to each of the traps
        DX=self.direction[0]/2*self.geometry.trap_sep
        DY=self.direction[1]/2*self.geometry.trap_sep
        # P is the vector that goes from the center to the colloid
        PX=self.colloid[0]
        PY=self.colloid[1]
        
        #Discriminant = self.colloid.dot(self.direction)
        #Discriminant = Discriminant/abs(Discriminant)
        
        #DX = DX*Discriminant
        #DY = DY*Discriminant
                
        W = self.geometry.stiffness*5e7
        
        ax1.plot(X,Y,'k')
        ax1.add_patch(patches.Circle(
            (X-DX,Y-DY),radius = W,
            ec='g', fc='g'))
        ax1.add_patch(patches.Circle(
            (X+DX,Y+DY),radius = W,
            ec='y', fc='y'))
        ax1.add_patch(patches.Circle(
            (X+PX,Y+PY), radius = W/3, ec='k', fc = 'none'))
        ax1.plot([X,X+PX],[Y,Y+PY],color='k')
        ax1.set_aspect("equal")
        
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
        if not (vector.__class__.__name__=="ndarray" \
            or vector.__class__.__name__=="list" \
            or vector.__class__.__name__=="tuple"):
                raise ValueError("The vector argument must be either a Vector, ndarray, list or tuple")
        # if vector is 2D, append zero
        if len(vector)==2:
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
        if vector.__class__.__name__=="ndarray":
            if vector.dot(self.direction)<0:
                cp = self.flip()
        elif vector.__class__.__name__=="list" or vector.__class__.__name__=="tuple":
            if np.array(vector).dot(self.direction)<0:
                cp = self.flip()
        return cp
        
class colloidal_ice(list):
    """ 
    The colloidal ice object is a list of colloid_in_trap objects.
    It also includes some extra parameters contained in the worldparams attribute. 
    It normally takes a spin ice object as input and generates one colloid_in_trap object for each spin
    """
    def __init__(self,*args,**kargs):

        self.worldparams = WorldParameters(**kargs)
        
        for i,s in enumerate(args[0]):
            """ ordering is a parameter of the SpinIce object. 
            It is important to pass it here to the colloid parameters because
            it affects how colloids are setup in the initial configuration"""
            c = colloid_in_trap(
                args[0][s].center,
                args[0][s].direction,
                InitialPosition = args[0].ordering,**kargs)
            
            self.append(c)

        self.worldparams.set_region(self)

    def __str__(self):
        
        PrntStr = ""
        for s in self:
            PrntStr += \
                self[s].__str__()
                
        return(PrntStr)
        
    def display(self, ax = False):
                
        if not ax:
            fig1, ax = plt.subplots(1,1)            

        for s in self:
            s.display(ax)

        plt.axis("equal")

    def framedata_to_colloids(self,frame_data,run):
        """ 
        Converts frame_data to a colloidal_ice object
        frame_data is a structured numpy array with the data from a single frame of a 
        colloidal ice simulation. This data 
        """
        number_of_runs = np.max(frame_data['type']);

        if (run+1)>=number_of_runs:
            raise ValueError("You are asking for a run that doesn't exist")
        
        """ Think about this part. 
            Trusting the order imposed by the LAMMPS sim sounds like it could be improved """
        frame_data.sort(0)
        X = frame_data['x'][frame_data['type']==run+1]
        Y = frame_data['y'][frame_data['type']==run+1]
        Z = frame_data['z'][frame_data['type']==run+1]

        Xc = frame_data['x'][frame_data['type']==number_of_runs]
        Yc = frame_data['y'][frame_data['type']==number_of_runs]
        Zc = frame_data['z'][frame_data['type']==number_of_runs]
        
        frame_centers = np.vstack([Xc.flatten(),Yc.flatten(),Zc.flatten()]).transpose()
        centers = np.array([np.array(c.center) for c in self])
        
        """ 
        This part finds the index of the frame_center array that corresponds to a place in the centers array
        The element where_in_frame[i] is the location of the framedata row corresponding to  colloidal_ice[i]
        """
        frame_center_tree = spa.cKDTree(frame_centers)
        where_in_frame = frame_center_tree.query_ball_point(centers/1000,10)
        
        for i,c in enumerate(self): 
            w = where_in_frame[i][0]
            c.colloid = np.array([X[w]-Xc[w],Y[w]-Yc[w],Z[w]-Zc[w]])*1000
            """ Ensure c.direction points to the colloid"""
            c.direction = c.direction*np.sign(np.sum(c.colloid*c.direction))
            
        return self
    
    def calculate_energy(self):
        """ Calculates the sum of the inverse cube of all the inter particle distances.
        For this it uses the spatial package of scipy which shifts this calculation to
        a compiled program and it's therefore faster.
        The energy output is given in 1/nm^3
        """
        colloids = np.array([np.array(c.center+c.colloid) for c in self])
        self.energy = sum(spa.distance.pdist(colloids)**(-3))
        return self.energy
