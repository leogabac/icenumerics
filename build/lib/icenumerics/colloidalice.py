import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import scipy.spatial as spa
import pandas as pd

from icenumerics.geometry import *
from icenumerics.parameters import *
from icenumerics.spins import *

from . import mc

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
               (tuple(self.colloid.magnitude)+tuple(self.center.magnitude)+tuple(self.direction)))
               
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
                
        W = (1/self.trap.stiffness.magnitude)*4e-3
        
        ax1.plot(X,Y,'k')
        ax1.add_patch(patches.Circle(
            (X-DX,Y-DY),radius = W,
            ec='g', fc='g'))
        ax1.add_patch(patches.Circle(
            (X+DX,Y+DY),radius = W,
            ec='y', fc='y'))
        ax1.add_patch(patches.Circle(
            (X+PX,Y+PY), radius = W/3, ec='k', fc = 'none'))
        #ax1.plot([X,X+PX],[Y,Y+PY],color='k')
        
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
        
        self.height_spread=height_spread
        self.susceptibility_spread = susceptibility_spread
                    
        self.extend(
            [colloid_in_trap(c,d,p,t) 
                for p,t,c,d in zip(particle,trap,centers,directions)])

        if region == None:
            units = centers[0].units
            lower_bounds = np.min(
                np.array([c.to(units).magnitude for c in centers])*units,0)
            upper_bounds = np.max(
                np.array([c.to(units).magnitude for c in centers])*units,0)
            
            region = np.vstack([lower_bounds,upper_bounds])*units
        
        self.region = region
        
        if periodic is None:
            periodic = False
            
        self.periodic = periodic
        
                
    def display(self, ax = None):
                
        if not ax:
            fig1, ax = plt.subplots(1,1)            

        for s in self:
            s.display(ax)
        
        ax.set_xlim([self.region[0,0].magnitude,self.region[1,0].magnitude])
        ax.set_ylim([self.region[0,1].magnitude,self.region[1,1].magnitude])
         
        ax.set_aspect("equal")
           
        #plt.axis("square")
    
    def pad_region(self,pad):
        self.region[0] = self.region[0]-pad
        self.region[1] = self.region[1]+pad
        
    def simulate(self, world, name,
        targetdir = '',
        include_timestamp = True,
        run_time = 60*ureg.s,
        framerate = 15*ureg.Hz,
        timestep = 100*ureg.us,
        output = ["x","y","z"]):
        
        particles = [c.particle for c in self]
        traps = [c.trap for c in self]

        colloids = [c.colloid.to(ureg.um).magnitude for c in self]*ureg.um
        centers = [c.center.to(ureg.um).magnitude for c in self]*ureg.um
        directions = [c.direction for c in self]
        
        # s = np.shape(np.array(colloids))
        # initial_randomization = np.random.randn(s[0],s[1])*0.1*ureg.um
        initial_displacement = np.array([[0,0,0.1]]*len(colloids))*ureg.um
        
        p_type = np.array(classify_objects(particles))
        t_type = np.array(classify_objects(traps))

        for p in np.unique(p_type):

            particles = mc.particles(
                colloids+centers+initial_displacement,
                atom_type = 0,
                atoms_id = np.arange(len(colloids)),
                radius = self[p].particle.radius,
                susceptibility = self[p].particle.susceptibility,
                drag = self[p].particle.drag)

        for t in np.unique(t_type):

            traps = mc.bistable_trap(
                centers,
                directions,
                particles,
                atom_type = 1, 
                trap_id = np.arange(len(centers))+len(colloids),
                distance = self[t].trap.trap_sep,
                height = self[t].trap.height,
                stiffness = self[t].trap.stiffness,
                height_spread = self.height_spread
                )
                
        world_sim = mc.world(
            particles,traps,
            region=self.region.transpose().flatten(),
            walls=[False,False,False],
            temperature = world.temperature,
            dipole_cutoff = world.dipole_cutoff,
            lj_cutoff = 0,
            lj_parameters = [0*ureg.pg*ureg.um**2/ureg.us**2,0],
            enforce2d = world.enforce2d,
            gravity = 0*ureg.um/ureg.us**2)
            
        field = mc.field(magnitude = world.field, 
            frequency = 0*ureg.Hz, angle = 0*ureg.degrees)
        
        self.sim = mc.sim(
            file_name = name, dir_name = targetdir, stamp_time = include_timestamp,
            timestep = timestep,
            framerate = framerate,
            total_time = run_time,
            output = output,
            particles = particles, traps = traps, world = world_sim, field = field)
        
        self.sim.generate_scripts()
        self.sim.run()
        
        self.trj = self.sim.load(read_trj = True)
        self.frames = self.trj.index.get_level_values("frame").unique()
        self.set_state_from_frame(frame = -1)
        
    def set_state_from_frame(self, frame):
        
        idx = pd.IndexSlice
        frame = self.frames[frame]
        for i,c in enumerate(self):
            c.colloid = self.trj.loc[idx[frame,i+1],["x","y","z"]].values*ureg.um - c.center
            c.direction = c.direction * np.sign(np.dot(c.colloid,c.direction))
    
    def calculate_energy(self):
        """ Calculates the sum of the inverse cube of all the inter particle distances.
        For this it uses the spatial package of scipy which shifts this calculation to
        a compiled program and it's therefore faster.
        The energy output is given in 1/nm^3
        """
        colloids = np.array([np.array(c.center+c.colloid) for c in self])
        self.energy = sum(spa.distance.pdist(colloids)**(-3))
        return self.energy
        
def classify_objects(object_list):
    """ Classifies objects by uniqueness. Returns a list with an object type directory."""
    o_type = -1

    def where(obj,obj_list):
        """returns the first occurence of `particle` in the array `particles`"""
        for i,o in enumerate(obj_list):
            if o==obj:
                return i

    type_dict = []

    for i,o in enumerate(object_list):
        loc = where(o,object_list[0:i])
        if loc is not None:
            type_dict.append(o_type)
        else:
            o_type = o_type+1
            type_dict.append(o_type)
    
    return type_dict