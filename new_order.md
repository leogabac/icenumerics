# What do I want to include in icenumerics.

Since most of my functions work on the trajectory dataframes, and pandas stuff. I think it is natural to include most of the stuff under `trajectory.py`
Furthermore, I want to create the `energy.py` submodule.

# Trajectory

## From auxiliary.py

def get_idx_from_position(centers,pos,tol=0.1):
def is_horizontal(direction):

Fix the position for PBC
def fix_position(position,a,size):

Classify vertices, in ROMAN NUMERALS
def classify_vertices(vrt):

Include the expanded vertices


def vrt_dict(path):
def vrt_counts(verticesDict):

Redo the averages function to be better, take into account the stuff from fig2 in stuckgs
def vrt_averages(counts,framerate):

Do all the vertices shenanigans
def do_vertices(params,data_path):

def trj2col(params,ctrj):

Count the vertices of a single frame df. I think is unnecessary
def vrtcount_sframe(vrt, column="type"):

This is unnecessary
def vrt_lastframe(path,last_frame=2399):

def vrt_at_frame(ctrj,frame):

def min_from_domain(f,domain):

Redo this
def positions_from_trj(ctrj):

Drop some columns for plotting purposes, try to pass list with the stuff to drop
def dropvis(ctrj):


def load_ctrj_and_vertices(params,data_path,size,realization = 1):

def get_rparalell(ctrj,particle,frame):

There should be a func for getting only the spin directions

See if there is a better way of doing this
def autocorrelation(ts):

I think this can be done better
def correlate_bframes(params,ts,sframes, stime= 0, etime = 60):


This just makes a bunch of integers from a list of strings
I could use simply np.asarray(list(map(int,x)))
def bint(x):

## From vertices.py
def trj2numpy(trj):
def numpy2trj(centers,dirs,rels):
def trj2trj(trj):

# Energy
def calculate_energy(dimensions,L,sel_particles):
def calculate_energy_noperiodic(dimensions,sel_particles):

Calculate energy with parallelized with taichi

# Square vertices
## From vertices.py
def create_lattice(a,N,spos=(0,0)):
def indices_lattice(vrt_lattice,centers,a,N):
def get_topological_charge_at_vertex(indices,dirs):
def get_charge_lattice(indices_lattice,dirs):
def dipole_lattice(centers,dirs,rels,vrt_lattice,indices_matrix):
def charge_op(charged_vertices):


