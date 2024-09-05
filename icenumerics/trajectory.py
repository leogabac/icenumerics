from icenumerics.spins import *
from icenumerics.colloidalice import colloidal_ice
from icenumerics.vertices import count_vertices
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial as sptl
import pandas as pd
from numba import jit,prange,float64,int64,complex128

def unwrap_trj(trj,bounds):
    """ Unwraps trj around periodic boundaries"""
    trj2 = trj.copy(deep=True)

    def unwrap(p):
        p.iloc[:] = np.unwrap(p,axis=0)
        return p

    for c in trj.columns:
        trj2[c] = (trj2[c] - bounds[c+"_min"].values)/(bounds[c+"_max"].values - bounds[c+"_min"].values)

    trj2 = (trj2*2*np.pi).groupby("id", group_keys=False).apply(unwrap)/(2*np.pi)

    for c in trj.columns:
        trj2[c] = trj2[c]*(bounds[c+"_max"].values - bounds[c+"_min"].values) + bounds[c+"_min"].values

    return trj2

def unwrap_frame(col_trj,bnd,axis):
    box_size = (bnd[axis+"_max"]-bnd[axis+"_min"]).values[0]
    mask = col_trj.loc[:,"c"+axis]>box_size/2
    col_trj.loc[mask,"c"+axis] = col_trj.loc[mask,"c"+axis]-box_size

    mask = col_trj.loc[:,"c"+axis]<-box_size/2
    col_trj.loc[mask,"c"+axis]= col_trj.loc[mask,"c"+axis]+box_size

def get_ice_trj(trj,bounds, atom_types = None, trap_types = None):
    """ Converts lammps trj to ice trj

    """
    # in the trj dataframe, traps and atoms are labeled by different types.
    # for single trap type, and single particle type, the default behaviour is to have traps as type 2 and atoms as type 1.
    # If the behaviour is not the default, it should be specified.
    if trap_types is None:
        trap_types = [2]

    try:
        traps = trj[trj.type.isin(trap_types)].copy(deep=True)
    except TypeError:
        traps = trj[trj.type.isin([trap_types])].copy(deep=True)

    try:
        atoms = trj[~trj.type.isin(trap_types)].copy(deep=True)
    except TypeError:
        atoms = trj[~trj.type.isin([trap_types])].copy(deep=True)

    traps = traps.rename(columns = {"mux":"dx","muy":"dy","muz":"dz"})
    atoms = unwrap_trj(atoms.filter(["x","y","z"]),bounds.loc[[0]])

    trj = []

    ## It turns out that the traps are not in the same order as the particles, when the system is bidisperse.
    # we first reindex the traps so that they start at zero, and increase consecutively
    traps_id = traps.index.get_level_values("id").unique()
    reindex_traps = pd.Series({t:i for i, t in enumerate(traps_id)})
    traps.reset_index(inplace = True)
    traps.id = traps.id.map(reindex_traps)
    traps = traps.set_index(["frame","id"]).sort_index()

    # we calculate the distance between traps and particles in the first frame,
    # and we build an atom index from the minimization of this distance
    distances = spa.distance.cdist(traps.loc[0,["x","y","z"]], atoms.loc[0,["x","y","z"]])
    reindex_atoms = pd.Series({a+1:t for a, t in enumerate(np.argmin(distances, axis = 0))})

    # now we reindex the atoms
    atoms.reset_index(inplace = True)
    atoms.id = atoms.id.map(reindex_atoms)
    atoms = atoms.set_index(["frame","id"]).sort_index()

    ## create a relative position vector. This goes from the center of the trap to the position of the particle
    colloids = atoms-traps
    colloids = colloids[["x","y","z"]]
    colloids.columns = ["cx","cy","cz"]

    colloids = colloids.dropna()

    traps = pd.concat([traps,colloids],axis=1)
    colloids = []
    atoms = []

    ## Flip those traps that are not pointing in the  direction of the colloids
    flip = np.sign((traps[["dx","dy","dz"]].values*traps[["cx","cy","cz"]].values).sum(axis=1))
    flip[flip==0]=1
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*flip[:,np.newaxis]

    ## make the direction vector unitary
    mag = np.sign((traps[["dx","dy","dz"]].values**2).sum(axis=1))
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*mag[:,np.newaxis]

    #timestep = 10e-3 #sec
    #traps["t"] = traps.index.get_level_values("frame")*timestep

    return traps

def get_ice_trj_single(col,i):

    lz_rd = col.sim.lazy_read
    trj = lz_rd[slice(i,i+1)]
    trj["t"] = trj.index.get_level_values("frame")*col.sim.timestep.to("sec").magnitude
    bnd = lz_rd.get_bounds(slice(i,i+1))

    traps = trj[trj.type==2].copy(deep=True)
    traps = traps.rename(columns = {"mux":"dx","muy":"dy","muz":"dz"})
    atoms = trj[trj.type==1].copy(deep=True)
    moments = atoms.filter(["mux", "muy", "muz"])
    atoms = atoms.filter(["x","y","z"])

    traps.loc[:,"id"] = traps.index.get_level_values("id").values
    traps.loc[:,"frame"] = traps.index.get_level_values("frame")
    traps.loc[:,"id"] = traps["id"]-min(traps["id"])+1
    traps = traps.set_index(["frame","id"])

    colloids = atoms-traps
    colloids = colloids[["x","y","z"]]
    colloids.columns = ["cx","cy","cz"]
    traps = pd.concat([traps,colloids, moments],axis=1)

    for ax in ["x","y","z"]:
        unwrap_frame(traps,bnd,ax)

    ## Flip those traps that are not pointing in the  direction of the colloids
    flip = np.sign((traps[["dx","dy","dz"]].values*traps[["cx","cy","cz"]].values).sum(axis=1))
    flip[flip==0]=1
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*flip[:,np.newaxis]

    ## make the direction vector unitary
    mag = np.sign((traps[["dx","dy","dz"]].values**2).sum(axis=1))
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*mag[:,np.newaxis]
    return traps.drop(columns="type"), bnd

def get_ice_trj_low_memory_hdf(col):
    import tqdm.notebook as tqdm
    name = os.path.split(col.sim.base_name)[1]
    mode = "a"
    col.sim.load(read_trj=False)

    for i,t in tqdm.tqdm(enumerate(col.sim.lazy_read.T),
                                total = len(col.sim.lazy_read.T),
                                desc = "Iterating through file",
                                leave = False ):

        trj, bnd = get_ice_trj_single(col,i)

        trj.astype("float16").to_hdf(
            os.path.join(col.dir_name,name+".h5"), key = "trj",
            mode = mode,
            format = "table", append = True)

        bnd.astype("float16").to_hdf(
            os.path.join(col.dir_name,name+".h5"), key = "bounds",
            mode = mode,
            format = "table", append = True)

        mode = "a"

def get_ice_trj_low_memory(col, dir_name = None):
    import tqdm.notebook as tqdm
    name = os.path.split(col.sim.base_name)[1]
    mode = "w"
    header = True
    col.sim.load(read_trj=False)

    if dir_name is None:
        for i,t in tqdm.tqdm(enumerate(col.sim.lazy_read.T),
                                    total = len(col.sim.lazy_read.T),
                                    desc = "Iterating through file" ):
            get_ice_trj_single(col,i)[0].to_csv(
                os.path.join(col.dir_name,name+".trj"), sep="\t",
                mode = mode, header = header)
            mode = "a"
            header = False
    else:
        for i,t in tqdm.tqdm(enumerate(col.sim.lazy_read.T),
                                    total = len(col.sim.lazy_read.T),
                                    desc = "Iterating through file" ):
            get_ice_trj_single(col,i)[0].to_csv(
                os.path.join(dir_name,name+".csv"),
                mode = mode, header = header)
            mode = "a"
            header = False          

def draw_frame(trj, frame_no = -1, region = None, radius = None, ax = None, sim = None, atom_type = 1, trap_type = 2, cutoff = None, trap_color = "blue", particle_color = "white"):
    
    # de donde viene el sim object?
    # intente con col.sim, pero diosito sabra de donde viene
    
    idx = pd.IndexSlice

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = (2,2), dpi = 150)

    if sim is not None:
        units =  sim.traps.cutoff.units

        region = [r.to(units).magnitude for r in sim.world.region]
        radius = sim.particles.radius.to(units).magnitude
        atom_type = sim.particles.atom_type+1
        trap_type = sim.traps.atom_type+1
        cutoff = sim.traps.cutoff.to(units).magnitude

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlim(region[0],region[1])
    ax.set_ylim(region[2],region[3])
    ax.set(aspect='equal')

    frames = trj.index.get_level_values("frame").unique()

    if "type" in trj.columns:
        atoms = trj[trj.type==atom_type]
        traps = trj[trj.type==trap_type]
    else:
        atoms = trj.loc[:,["x","y","z"]]+trj.loc[:,["cx","cy","cz"]].values
        try:
            trj = trj.drop(columns={"mux","muy","muz"})
        except:
            pass
        traps = trj.rename(columns = {"dx":"mux","dy":"muy","dz":"muz"})

    patches = []

    for i,t in traps.loc[idx[frames[frame_no],:],:].iterrows():
        
        c = plt.Circle(
            (t.x+t.mux/2,t.y+t.muy/2), cutoff,color = trap_color)
        patches.append(c)
        c = plt.Circle(
            (t.x-t.mux/2,t.y-t.muy/2), cutoff,color = trap_color)
        patches.append(c)
        width = t.mux+2*np.abs(cutoff*(not np.abs(t.muy)<1e-10))
        height = t.muy+2*np.abs(cutoff*(not np.abs(t.mux)<1e-10))
        c = plt.Rectangle(
            (t.x-width/2,t.y-height/2),
            width = width, height = height,color = trap_color)
        patches.append(c)

    for i,a in atoms.loc[idx[frames[frame_no],:],:].iterrows():
        c = plt.Circle((a.x,a.y), radius, facecolor = particle_color, edgecolor = "black")
        patches.append(c)

    for p in patches:
        ax.add_patch(p)
    return patches

def animate(trj, sl = slice(0,-1,1), region = None, radius = None, ax = None, sim = None, atom_type = 1, trap_type = 2, cutoff = None, framerate = None, verb=False, start=0, end=False, step = 1, speedup = 1, preserve_limits = False):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = (2,2), dpi = 150)
    fig = ax.figure

    if sim is not None:

        units =  sim.traps.cutoff.units
        region = [r.to(units).magnitude for r in sim.world.region]
        radius = sim.particles.radius.to(units).magnitude
        framerate = sim.framerate.magnitude
        timestep = sim.timestep.magnitude
        atom_type = sim.particles.atom_type+1
        trap_type = sim.traps.atom_type+1
        cutoff = sim.traps.cutoff.to(units).magnitude
        if cutoff == np.inf:
            cutoff=radius*1.1
    if cutoff is None:
        cutoff=radius*1.1

    if not preserve_limits:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim(region[0],region[1])
        ax.set_ylim(region[2],region[3])
        ax.set(aspect='equal')

    frames = trj.index.get_level_values("frame").unique()
    frames = frames[sl]

    dt_video = 1000/framerate/speedup # video timestep in miliseconds


    if "type" in trj.columns:
        atoms = trj[trj.type==atom_type]
        traps = trj[trj.type==trap_type]
    else:
        atoms = trj.loc[:,["x","y","z"]]+trj.loc[:,["cx","cy","cz"]].values
        traps = trj.rename(columns = {"dx":"mux","dy":"muy","dz":"muz"})

    patches = []

    atom_patches = []
    trap_patches = []

    for i,a in atoms.loc[idx[frames[0],:],:].iterrows():
        c = plt.Circle((0,0), radius, facecolor = "white", edgecolor = "black")
        atom_patches.append(c)

    for i,t in traps.loc[idx[frames[0],:],:].iterrows():

        c = plt.Circle(
            (t.x+t.mux/2,t.y+t.muy/2), cutoff,color = "blue")
        trap_patches.append(c)

        c = plt.Circle(
            (t.x-t.mux/2,t.y-t.muy/2), cutoff,color = "blue")
        trap_patches.append(c)
        width = t.mux+2*np.abs(cutoff*(not np.abs(t.muy)<1e-10))
        height = t.muy+2*np.abs(cutoff*(not np.abs(t.mux)<1e-10))
        c = plt.Rectangle(
            (t.x-width/2,t.y-height/2),
            width = width, height = height,color = "blue")
        trap_patches.append(c)

    def init():
        for t in trap_patches:
            ax.add_patch(t)
        for a in atom_patches:
            ax.add_patch(a)
        return trap_patches+atom_patches


    def update(frame):
        if verb:
            print("frame[%u] is "%frame,frames[frame])

        for i,((f,ind),atom)in enumerate(atoms.loc[idx[frames[frame],:],:].iterrows()):
            atom_patches[i].center = (atom.x,atom.y)
            #print(atom_patches[i].center)
        for a in atom_patches:
            ax.add_patch(a)

        return atom_patches

    anim = mpl.animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len(frames), interval=int(round(dt_video)), blit=True);
    plt.close(anim._fig)

    return anim


# since there is no particular order, I will start here
@jit(nopython=True)
def get_idx_from_position(centers,pos,tol=0.1):
    """
        Get the index in the centers array from a position vector.
        ----------
        * centers: centers of the traps
        * pos: np array with a 3D coordinate
    """

    for i,center in enumerate(centers):
        distance = np.linalg.norm(center - pos)
        if np.isclose(0,distance,atol=tol):
            return i

def is_horizontal(direction):
    """
        Checks if a given direction is horizontal.
        ----------
        Parameters:
        * direction
    """
    x = np.array([1,0,0])
    x_dot_dir = np.dot(direction,x)

    return isclose(abs(x_dot_dir),1,rel_tol=1e-3)

@jit(nopython=True)
def fix_position(position,a,size):
    """
        Fixes the position to fit in the box
        0 < x < size*a, and
        0 < y < size*a
        ----------
        Parameters:
        * position: Position vector in 3D
        * a: lattice constant
        * size: size of the system
    """
    L = size*a

# Apply BC to x
    position[0] = position[0] % L
    if position[0] < 0:
        position[0] += L

# Apply BC to y
    position[1] = position[1] % L
    if position[1] < 0:
        position[1] += L

    return position

def classify_vertices(vrt):
    """
        Classifies the vertices in I, II, III, IV, V, VI types.
        Returns a DataFrame
        ----------
        Parameters:
        * vrt (pd Dataframe): Vertices df
    """

    vrt["type"] = ['tmp'] * len(vrt)

    vrt.loc[vrt.eval("coordination==4 & charge == -4"),"type"] = 'I'
    vrt.loc[vrt.eval("coordination==4 & charge == -2"),"type"] = 'II'
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)==0"),"type"] = 'III'
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)>0"),"type"] = 'IV'
    vrt.loc[vrt.eval("coordination==4 & charge == 2"),"type"] = 'V'
    vrt.loc[vrt.eval("coordination==4 & charge == 4"),"type"] = 'VI'
    return vrt

def vrt_dict(vrt_path):
    """
        Walks path and imports all DFs into a Dictionary, classifies the vertices and drops boundaries.
        Returns a dictionary with all the DataFrames.
        ----------
        Parameters:
        * vrt_path: Path where the vertices are located.
    """

    # get a list of the filenames
    all_files = os.listdir(vrt_path)
    trj_files = [file for file in all_files if file.startswith('vertices')]

    vrt_dict = {} # Initialize
    realizations = len(trj_files)

    # loop all the files and classify
    for i,file in enumerate(trj_files):
        file_path = os.path.join(vrt_path,file)

        vrt = pd.read_csv(file_path, index_col=['frame','vertex'])
        vrt = classify_vertices(vrt)

        # drop nan and tmp
        vrt = vrt.dropna()
        vrt = vrt[vrt['type']!='tmp']
        vrt_dict[i+1] = vrt
    return vrt_dict

def vrt_counts(vertices_dict):
    """
        Loops the vrt_dict with all realiztions and gets the counts for vertex type
        Returns a dictionary with the counts DF for all experiments
        ----------
        Parameters:
        * vertices_dict: Dictionary from vrt_dict()
    """

    return {i:count_vertices(vertices) for i,vertices in vertices_dict.items()}

def vrt_averages(data_path, max_frame=6000):
    """
        Gets the averages for all realizations.
        ----------
        Parameters:
        * data_path
        * max_frame (optional): in case we only want a subset of vertices.
    """
    vertices = vrt_dict(data_path)
    counts = vrt_counts(vertices)

    # take all the fractions
    # concat in a df
    # take average
    # get until max frame
    mean_counts = pd.concat(
        [stuff.fraction for stuff in counts.values()],axis=1
    ).dropna().mean(axis=1).reset_index().query(f'frame<={max_frame}').set_index(['frame','type'])

    return mean_counts

def unstack(df,col_names = ['I','II','III','IV','V','VI']):
    """
        Unstacks the vertices counts so that we have types as columns and frames as rows.
        In this way we can have accessible timeseries for the vertices.
        ----------
        Parameters:
        * df: result from vrt_averages()
        * col_names (optional)
    """
    # try to at some point do this automatically
    frames = df.index.get_level_values('frame').unique().to_list()
    unstacked = pd.DataFrame(df.unstack(level='type').to_numpy(),columns=col_names)
    unstacked['frame'] = frames
    return unstacked.set_index('frame')

def test():
    print('oli')
