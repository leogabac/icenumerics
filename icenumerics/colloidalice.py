import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.animation as anm
import scipy.spatial as spa
import pandas as pd
import warnings

from icenumerics.geometry import *
from icenumerics.parameters import *
from icenumerics.spins import *

from . import mc

class colloid_in_trap():
    """
    An object 'colloid_in_trap' represents a colloidal particle in a bistable trap. It has three main properties:
    
    * center is the center of the bistable trap. 
    * direction is a vector (whose magnitude is not important) that points from one stable position to the other.
    * colloid is a vector that indicates where the colloid is placed with respect to the center.
    
    Each of these quantities are represented by lists of three elements, which correspond to 
    vectors in R3 space. A colloid_in_trap object also has the properties:
    
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
    def __init__(self, center, direction, particle, trap):
    
        self.center = np.array(center.magnitude,dtype="float")*center.units
        
        # Direction is always unitary
        self.direction = np.array(direction.magnitude,dtype="float")/np.linalg.norm(direction.magnitude,2)
        
        self.particle = particle
        self.trap = trap
                
        """ Colloid position is given by the direction vectors"""
        self.colloid = self.direction * self.trap.trap_sep/2
        
    def __str__(self):
        """ Prints a string which represents the colloid_in_trap """
        
        return("Colloid is in [%d %d %d], trap is [%d %d %d %d %d %d]\n" %\
               (tuple(self.colloid.magnitude)+tuple(self.center.magnitude)+tuple(self.direction)))
               
    def display(self, ax1=False, units = None, scale = 1):
        """ Draws a figure with the trap and the colloid inside it"""
        if not ax1:
            fig, ax1 = plt.subplots(1,1)
            
        patches = self.create_patch(units = units, scale=scale)
        
        #ax1.plot(X,Y,'k')
        ax1.add_patch(patches[0])
        ax1.add_patch(patches[1])
        ax1.add_patch(patches[2])
        #ax1.plot([X,X+PX],[Y,Y+PY],color='k')
        
    def create_patch(self, units = None, scale = 1):
        """ Draws a figure with the trap and the colloid inside it"""

        if not units:
            units = self.center.units
            
        X=self.center[0].to(units).magnitude
        Y=self.center[1].to(units).magnitude
        
        # D is the vector that goes fom the center to each of the traps
        DX=self.direction[0]/2*self.trap.trap_sep.to(units).magnitude
        DY=self.direction[1]/2*self.trap.trap_sep.to(units).magnitude
        # P is the vector that goes from the center to the colloid
        PX=self.colloid[0].to(units).magnitude
        PY=self.colloid[1].to(units).magnitude

        W = (self.particle.radius.to(units).magnitude)

        return [ptch.Circle((X-DX,Y-DY), radius = W*scale, ec='g', fc='g'),
                ptch.Circle((X+DX,Y+DY), radius = W*scale, ec='y', fc='y'),
                ptch.Circle((X+PX,Y+PY), radius = W, ec='k', fc = 'none')]

    def update_patch(self, patch, units = None):
        """ Changes the configuration of the colloid display"""
        if not units:
            units = self.center.units
        
        X=self.center[0].to(units).magnitude
        Y=self.center[1].to(units).magnitude
        
        # D is the vector that goes fom the center to each of the traps
        DX=self.direction[0]/2*self.trap.trap_sep.to(units).magnitude
        DY=self.direction[1]/2*self.trap.trap_sep.to(units).magnitude
        # P is the vector that goes from the center to the colloid
        PX=self.colloid[0].to(units).magnitude
        PY=self.colloid[1].to(units).magnitude
        
        patch[0].center = (X-DX,Y-DY)
        patch[1].center = (X+DX,Y+DY)
        patch[2].center = (X+PX,Y+PY)
    
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
    The colloidal ice object is a list of colloid_in_trap objects. It also includes some extra parameters contained in the worldparams attribute. It normally takes a spin ice object as input and generates one colloid_in_trap object for each spin.
    The arrangement parameter defines the positions and directions of the colloidal ice. There are two possible inputs:
    
        * a `spins` object: in this case the colloidal ice is copied from the spins arrangement. 
        * a `dict` object: this `dict` object must contain two arrays, `center` and `direction`.
    
    `particle` and `trap` are parameter containers created with the `particle` and `trap` generators. They can be a single object, or a list. If it is a list, it must coincide with the number of elements defined by the `arrangement` parameter.
    """
    def __init__(self, arrangement, particle, trap, height_spread = 0, susceptibility_spread = 0, region = None, periodic = None):

        
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
                np.array([c.to(units).magnitude for c in centers]),0)
            upper_bounds = np.max(
                np.array([c.to(units).magnitude for c in centers]),0)
            
            region = np.vstack([lower_bounds,upper_bounds])*units
        
        self.region = region
        
        if periodic is None:
            periodic = False
            
        self.periodic = periodic
        
    def display(self, ax = None, scale = 1):
                
        if not ax:
            fig1, ax = plt.subplots(1,1)   
            
        units = self.region.units

        for s in self:
            s.display(ax,units, scale=scale)
        
        ax.set_xlim([self.region[0,0].magnitude,self.region[1,0].magnitude])
        ax.set_ylim([self.region[0,1].magnitude,self.region[1,1].magnitude])
         
        ax.set_aspect("equal")
           
    def animate(self,sl=slice(0,-1,1),ax=None,speed = 1, verb=False):
        """ Animates a trajectory """
    
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=(7,7))
        else:
            fig = ax.figure
        
        region = [r.magnitude for r in self.sim.world.region]
        len_units = self.sim.world.region.units
        time_units = self.sim.total_time.units
        
        particles = self.trj.index.get_level_values('id').unique()
        n_of_particles = len(particles)
        frames = self.trj.index.get_level_values('frame').unique().values

        region = [r.magnitude for r in self.sim.world.region]
        try:
            radius = [self.sim.particles.radius.to(len_units).magnitude]
        except AttributeError:
            radius = [p.radius.to(len_units).magnitude for p in self.sim.particles]
        
        framerate = self.sim.framerate.to(1/time_units).magnitude
        runtime = self.sim.total_time.to(time_units).magnitude
        timestep = self.sim.timestep.to(time_units).magnitude
        frame_duration = (self.sim.total_time/(len(frames-1))*sl.step).to(ureg.ms).magnitude/speed
        
        patches = [p for c in self for p in c.create_patch(len_units)]
        
        def init():
            for patch in patches:
                ax.add_patch(patch)
            return patches

        def animate(frame_id):
            frame = frames[sl][frame_id]
            self.set_state_from_frame(frame = frame)
            if verb:
                print("frame[%u] is "%frame,frames[frame])
            for p1,p2,p3,c in zip(patches[0::3],patches[1::3],patches[2::3],self):
                c.update_patch([p1,p2,p3],len_units)
            for patch in patches:
                ax.add_patch(patch)
            return patches

        ax.set_xlim(region[0],region[1])
        ax.set_ylim(region[2],region[3])
        ax.set(aspect='equal')
        
        anim = anm.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(frames[sl]), interval=frame_duration, blit=True)
        plt.close(anim._fig)
            
        return anim
        
    def pad_region(self,pad,enforce2d=True):
        
        self.region[0] = self.region[0]-pad
        self.region[1] = self.region[1]+pad
        
        if enforce2d:
            self.region[:,2] = np.array([-.02,.02])*ureg.um
            
    def simulate(self, *args, **kargs):
        
        self.simulation(*args,**kargs)
        
        self.run_simulation()
        
        self.load_simulation()
    
    def simulation(self, world, name, targetdir = '', include_timestamp = True, run_time = 60*ureg.s, framerate = 15*ureg.Hz, timestep = 100*ureg.us, output = ["x","y","z"], processors = 1):
        particles = [c.particle for c in self]
        traps = [c.trap for c in self]

        colloids = [c.colloid.to(ureg.um).magnitude for c in self]*ureg.um
        centers = [c.center.to(ureg.um).magnitude for c in self]*ureg.um
        directions = np.array([c.direction for c in self])
        
        # s = np.shape(np.array(colloids))
        # initial_randomization = np.random.randn(s[0],s[1])*0.1*ureg.um
        initial_displacement = np.array([[0,0,0.001]]*len(colloids))*ureg.um
        positions = colloids+centers+initial_displacement
        
        p_type, inverse_p = unique_objects(particles)
        t_type, inverse_t = unique_objects(traps)
        
        p_id = np.arange(len(colloids))
        
        particle_types = []
        
        particle_locations = []
        
        for i,p in enumerate(p_type):
            
            particle_types.append(mc.particles(
                positions[np.array(inverse_p)==i],
                atom_type = 0,
                atoms_id = np.arange(len(colloids)),
                radius = p.radius,
                susceptibility = p.susceptibility,
                drag = p.drag))
                
            particle_locations.append(p_id[np.array(inverse_p)==i])
        
        particle_locations = np.array([loc for p_loc in particle_locations for loc in p_loc])    
        self.particle_locations = particle_locations
        
        trap_types = []
        # trapbonds

        part_in_type = [np.where(np.array(inverse_p)==t)[0] for t, pt in enumerate(p_type)]
        trap_bond_i = lambda i,p: np.concatenate([[p], list(np.where(part_in_type[p]==i)[0])])
        trap_bonds = [trap_bond_i(i,p) for i, p in enumerate(inverse_p)]
        
        for i,t in enumerate(t_type):
            
            ## inverse_t relates a type of trap, to a position in the trap array
            ## inverse_p relates a type of particle to a position in the particle array
            
            ## the particle array and the trap array have the same order: namely, particle i should bond with trap i. 
            
            subsets = [trap_bonds[j] for j, typ in enumerate(inverse_t) if typ==i]
            
            trap_types.append(mc.bistable_trap(
                centers[np.array(inverse_t)==i],  
                directions[np.array(inverse_t)==i],
                particle_types, 
                subsets = subsets,
                # I don't know how this would work for several types of particles. 
                atom_type = 1, 
                trap_id = np.arange(len(centers))+len(colloids),
                distance = t.trap_sep,
                height = t.height,
                stiffness = t.stiffness,
                height_spread = self.height_spread,
                cutoff = t.cutoff))
            
        world_sim = mc.world(
            particle_types,trap_types,
            region=self.region.transpose().flatten(),
            walls=[False,False,False],
            boundaries =  world.boundaries,
            temperature = world.temperature,
            dipole_cutoff = world.dipole_cutoff,
            lj_cutoff = 0,
            lj_parameters = [0*ureg.pg*ureg.um**2/ureg.us**2,0],
            enforce2d = world.enforce2d,
            gravity = 0*ureg.um/ureg.us**2)
            
        field = mc.field(magnitude = world.field, 
            frequency = 0*ureg.Hz, angle = 0*ureg.degrees)
        
        self.run_params = {
            "file_name":name,
            "dir_name":targetdir,
            "stamp_time":include_timestamp,
            "timestep":timestep,
            "framerate":framerate,
            "total_time":run_time,
            "output":output,
            "particles":particle_types,
            "traps":trap_types,
            "world":world_sim,
            "field":field,
            "processors":processors}
            
            
        self.name = name
        self.dir_name = targetdir
        self.include_timestamp = include_timestamp
        self.sim = mc.sim(**self.run_params)
    
    def update_simulation(self):
        self.sim = mc.sim(**self.run_params)
        
    def run_simulation(self):
        self.sim.generate_scripts()
        self.sim.run()
    
    def load_simulation(self, sl = slice(0,-1,1)):
        """ Loads the results from a simulation from the .lammpstrj file."""
        self.trj = self.sim.load(read_trj = True, sl = sl)
        self.bnd = self.sim.lazy_read.get_bounds(sl = sl)
        self.frames = self.trj.index.get_level_values("frame").unique()
        self.set_state_from_frame(frame = -1)
            
    def set_state_from_frame(self, frame):
        
        frame = self.frames[frame]
        index = np.argsort(self.particle_locations)
        
        for i,c in enumerate(self):

            c.colloid = self.trj.loc[(frame,index[i]+1),["x","y","z"]].values*ureg.um - c.center

            dot_prod = np.dot(c.colloid.magnitude,c.direction)
            dot_prod_sign = (dot_prod>=0)*1+(dot_prod<0)*(-1)
            
            c.direction = c.direction * dot_prod_sign
            
        return self
    
    def calculate_energy(self, B = 1*ureg.mT):
        """ Calculates the sum of the inverse cube of all the inter particle distances.
        For this it uses the spatial package of scipy which shifts this calculation to
        a compiled program and it's therefore faster.
        The energy output is given in 1/nm^3
        ------
        Parameters:
        B (Quantity): The physical parameters of the particles ($\chi$, radius) are stored in the `col` object, but the field needs to be specified. The `calculate_energy` method accepts a vector field, or a scalar quantity. If a scalar quantity is given, it is asumed to be in the vertical direction. 
        
        Results: 
        U (Quantity): The total dipole-dipole energy stored in the system.
        
        todo: it might be useful to be able to calculate the total internal energy, including the energy of the traps. 
        """
        
        
        mu0 = 4e5*np.pi*ureg.pN/ureg.A**2
        positions = np.array([np.array(c.center.to("um").magnitude+c.colloid.to("um").magnitude) for c in self])
        distances = spa.distance.pdist(positions)
        
        moment = lambda part: 4/3*np.pi*part.radius**3 * B * part.susceptibility / mu0
        moments = np.array([moment(c.particle).to(ureg.A*ureg.um**2).magnitude for c in self])
        
        try:
            # This should fail if B is not a vector
            
            pairs = np.array([[i,j+i+1] for i,p1 in enumerate(positions[:]) for j,p2 in enumerate(positions[i+1:])])
            r = positions[pairs[:,0]]-positions[pairs[:,1]]
            mdotm = np.sum(moments[pairs[:,0]]*moments[pairs[:,1]],axis=1)
            m1dotr = np.sum(r*moments[pairs[:,0]],axis=1)
            m2dotr = np.sum(r*moments[pairs[:,1]],axis=1)
            
            U = sum( -mu0.magnitude / (4*np.pi) * ((3*m1dotr*m2dotr)/distances**5-mdotm/distances**3))
            
        except np.AxisError:
            
            mdotm = np.array([m1*moments[j+i+1] 
                            for i,m1 in enumerate(moments[:]) 
                            for j,m2 in enumerate(moments[i+1:])])
                
            U = sum(mu0.magnitude / (4*np.pi) * mdotm * (distances)**(-3))
 
        
        self.energy = (U*ureg.pN*ureg.um).to("pN nm")
        
        return self.energy
    
    def DataFrame(self):
        frames = self.trj.groupby("frame").count().index

        
        return pd.concat([pd.DataFrame(data = np.array(
            [np.concatenate([c.center.magnitude,c.direction,c.colloid]) 
             for c in self.set_state_from_frame(f)]),
                     columns = ["x","y","z","dx","dy","dz","cx","cy","cz"],
                     index = pd.Index(range(len(self)),name = "id"))
                  for f in frames], keys = frames, names = ["frame"])
        
    def where(self,center,tol=None):
    
        if tol is None:
            tol = 1e-6*ureg.um
            
        return [i for i,c in enumerate(self) if np.linalg.norm(c.center-center)<tol]
    
    def copy(self,deep = False):
        import copy as cp
        if deep:
            return cp.deepcopy(self)
        else:
            return cp.copy(self)

    def randomize(self):

        import random

        for c in self:
            if random.randint(0, 1):
                c.colloid = -c.colloid
                c.direction = -c.direction
                
    def remove(self, idx):
        """ removes a list of colloids given by the indices idx"""
        col_list = [c for c in self]
        self.clear()
        self.extend([c for i,c in enumerate(col_list) if i not in idx])

def unique_objects(object_list):
    """ Classifies objects by uniqueness. 
    Returns:
    list_of_unique: a list with a single instance of each unique objects
    inverse: an array of where in list_of_unique the object is found. list_of_unque[inverse] == object_list
    """
    
    set_of_unique = {t for t in object_list}
    list_of_unique = list(set_of_unique)
    
    
    def where(obj, list_of_unique):
        """ returns the location of the object obj in the list_of_unique. """
        for i,unique_object in enumerate(list_of_unique):
            if obj == unique_object:
                return i
            
    inverse = [where(obj,list_of_unique) for obj in object_list]
     
    return list_of_unique, inverse

def classify_objects(object_list):
    """ Classifies objects by uniqueness. Returns a list with an object type directory."""
    o_type = -1 # Default (maybe)

    def where(obj,obj_list):
        """returns the first occurence of `particle` in the array `particles`"""
        for i,o in enumerate(obj_list):
            if o==obj:
                return i
                
    # This warning was made in 25-05-2020.
    warnings.warn(DeprecationWarning("The function classify_objects, in colloidalice doesn't work properly and will be removed soon. Use instead the function unique_objects"))
    
    type_dict = []

    for i,o in enumerate(object_list):
        loc = where(o,object_list[0:i])
        if loc is not None:
            type_dict.append(o_type)
        else:
            o_type = o_type+1
            type_dict.append(o_type)
    
    return type_dict
