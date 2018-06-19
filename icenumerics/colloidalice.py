import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import scipy.spatial as spa

from icenumerics.geometry import *
from icenumerics.parameters import *
from icenumerics.spins import *

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
        center, direction, particle, trap):
        """ initializes a colloid_in_trap object """
        
        self.center = np.array(center)*center.units
        
        # Direction is always unitary
        self.direction = np.array(direction)/np.linalg.norm(direction.magnitude,2)
        
        self.particle = particle
        self.trap = trap
                
        """ Colloid position is given by the direction vectors"""
        self.colloid = self.direction * self.trap.trap_sep/2
        
    def __str__(self):
        """ Prints a string which represents the colloid_in_trap """
        return("Colloid is in [%d %d %d], trap is [%d %d %d %d %d %d]\n" %\
               (tuple(self.colloid)+tuple(self.center)+tuple(self.direction)))
               
    def display(self, ax1=False):
        """ Draws a figure with the trap and the colloid inside it"""
        if not ax1:
            fig, ax1 = plt.subplots(1,1)
            
        X=self.center[0].magnitude
        Y=self.center[1].magnitude
        # D is the vector that goes fom the center to each of the traps
        DX=self.direction[0]/2*self.trap.trap_sep.to(self.center[0].units).magnitude
        DY=self.direction[1]/2*self.trap.trap_sep.to(self.center[1].units).magnitude
        # P is the vector that goes from the center to the colloid
        PX=self.colloid[0].to(self.center[0].units).magnitude
        PY=self.colloid[1].to(self.center[1].units).magnitude
        
        #Discriminant = self.colloid.dot(self.direction)
        #Discriminant = Discriminant/abs(Discriminant)
        
        #DX = DX*Discriminant
        #DY = DY*Discriminant
                
        W = self.trap.stiffness.magnitude*5e4
        
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
    def __init__(self,
    arrangement,particle,trap,
    height_spread = 0, susceptibility_spread = 0,
    region = None, periodic = None):
        """ 
        The arrangement parameter defines the positions and directions of the colloidal ice. There are two possible inputs:
            * a `spins` object: in this case the colloidal ice is copied from the spins arrangement. 
            * a `dict` object: this `dict` object must contain two arrays, `center` and `direction`.
        `particle` and `trap` are parameter containers created with the `particle` and `trap` generators. They can be a single object, or a list. If it is a list, it must coincide with the number of elements defined by the `arrangement` parameter.
        """
        
        if arrangement.__class__.__name__ == "spins":
            centers = [s.center for s in arrangement]
            directions = [s.direction for s in arrangement]
            
        else:
            centers = arrangement['centers']
            directions = arrangement['directions']
        
        if not hasattr(particle,'__getitem__'):
            particle = [particle for c in centers]
        if not hasattr(trap,'__getitem__'):
            trap = [trap for c in centers]
        
        height_disorder = np.random.randn(len(trap))*height_spread
        susceptibility_disorder = np.random.randn(len(trap))*susceptibility_spread
            
        for t,p,hdis,sdis in zip(
                trap,particle,height_disorder,susceptibility_disorder):
            t.height = t.height*hdis
            p.susceptibility = p.susceptibility*sdis
                    
        self.extend(
            [colloid_in_trap(c,d,p,t) 
                for p,t,c,d in zip(particle,trap,centers,directions)])

        if region == None:
            units = centers[0].units
            lower_bounds = np.min(
                np.array([c.to(units).magnitude for c in centers])*units,0)
            upper_bounds = np.max(
                np.array([c.to(units).magnitude for c in centers])*units,0)
            
            region = np.vstack([lower_bounds,upper_bounds])
        
        self.region = region
        
        if periodic is None:
            periodic = False
            
        self.periodic = periodic
        
                
    def display(self, ax = None):
                
        if not ax:
            fig1, ax = plt.subplots(1,1)            

        for s in self:
            s.display(ax)
        
        ax.set_xlim([self.region[0,0],self.region[1,0]])
        ax.set_ylim([self.region[0,1],self.region[1,1]])
         
        ax.set_aspect("equal")
           
        #plt.axis("square")
    
    def pad_region(self,pad):
        self.region[0] = self.region[0]-pad
        self.region[1] = self.region[1]+pad
        
    def simulate(self,
        world,
        name,
        targetdir = '',
        include_timestamp = True,
        run_time = 60*ureg.s,
        framerate = 15*ureg.Hz,
        timestep = 10*ureg.us,
        ):
        
        
        
        pass

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
