from icenumerics.spins import *
from icenumerics.colloidalice import colloidal_ice
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial as sptl
import pandas as pd

import tqdm.notebook as tqdm

def spin_crossing_point(S1,S2):
    # This works well in 2d. In 3d it's triciker
    if not (S1['Direction']==S2['Direction']).all():
        A = np.ones([2,2])
        A[:,0] = S1['Direction']
        A[:,1] = -S2['Direction']

        b = np.array([
            S2['Center'][0]-S1['Center'][0],
            S2['Center'][1]-S1['Center'][1]])

        lam = np.linalg.solve(A,b)

        return S1['Center']+lam[0]*S1['Direction']
    else:
        return np.Inf+np.zeros(np.shape(S1['Center']))

def unique_points(points,tol = 0.1):
    """Returns only the distinct points (with a tolerance)."""
    flatten = lambda lst: [el for l in lst for el in l]

    unique_points = []
    inverse = np.empty(len(points), dtype="uint16")
    copies_assigned = []

    for i,p in enumerate(points):
        if not np.isin(i, flatten(copies_assigned)):

            kdt_points = spa.cKDTree(points)

            same_point_copies = kdt_points.query_ball_point(p, tol)

            copies_assigned.append(same_point_copies)
            unique_points.append(points[same_point_copies].mean(axis=0))
            inverse[same_point_copies] = len(unique_points)-1

    unique_points = np.array(unique_points)
        
    return unique_points, inverse, copies_assigned
    
def colloidal_ice_vector(C):
    """Extracts an array of centers and directions from a Colloidal Ice System"""
    Vectors = np.array(np.zeros(len(C)),dtype=[('Center',np.float,(2,)),('Direction',np.float,(2,))])
    i=0
    for c in C:
        Vectors[i] = (c.center[0:2].magnitude,c.direction[0:2])
        i=i+1
    return Vectors
    
def spin_ice_vector(S):
    """Extracts an array of centers and directions from a Spin Ice System"""
    Vectors = np.array(np.zeros(len(S)),dtype=[('Center',np.float,(2,)),('Direction',np.float,(2,))])
    i=0
    for s in S:
        Vectors[i] = (s.center[0:2],s.direction[0:2])
        i=i+1
    return Vectors
    
def trj_ice_vector(trj_frame):
    """Extracts an array of centers and directions from a frame in a trj"""
    
    Vectors = np.array(np.zeros(len(trj_frame)),dtype=[('Center',np.float,(2,)),('Direction',np.float,(2,))])
    Vectors["Center"] = trj_frame.loc[:,["x","y"]]
    
    d = np.sqrt((trj_frame.loc[:,["dx","dy"]]**2).sum(axis=1))
    Vectors["Direction"] = trj_frame[["dx","dy"]].div(d, axis=0)
    
    return Vectors
        
def calculate_neighbor_pairs(Centers):
    """This function makes a list of all the Pairs of Delaunay Neighbors from an array of points"""
    
    tri = sptl.Delaunay(Centers)

    # List all Delaunay neighbors in the system
    NeighborPairs = np.array(np.zeros(2*np.shape(tri.simplices)[0]),
                             dtype=[('Pair',np.int,(2,)),('Distance',np.float),('Vertex',np.float,(2,))])

    i = 0
    for t in tri.simplices:
        NeighborPairs[i]['Pair'] = np.sort(t[0:2])
        NeighborPairs[i]['Distance'] = sptl.distance.euclidean(Centers[t[0]],Centers[t[1]])
        NeighborPairs[i+1]['Pair'] = np.sort(t[1:3])
        NeighborPairs[i+1]['Distance'] = sptl.distance.euclidean(Centers[t[1]],Centers[t[2]])
        i = i+2

    return NeighborPairs

def from_neighbors_get_nearest_neighbors(NeighborPairs):
    # This function takes a list of Delaunay Neighbor Pairs and returns only those which are close to the minimum distance.
    NeighborPairs['Distance']=np.around(NeighborPairs['Distance'],decimals=4)
    NeighborPairs = NeighborPairs[NeighborPairs['Distance']<=np.min(NeighborPairs['Distance'])*1.1]
    
    return NeighborPairs

def get_vertices_positions(NeighborPairs,spins):
    # From a list of Spins, get neighboring spins, and get the crossing point of each, which defines a vertex.  
    for i,n in enumerate(NeighborPairs):
        NeighborPairs[i]['Vertex'] = spin_crossing_point(spins[n['Pair'][0]],spins[n['Pair'][1]])[0:2]
    
    return NeighborPairs

def ice_to_spins(ice, id_label=None):

    if ice.__class__.__name__ == "colloidal_ice":
        spins = colloidal_ice_vector(ice)
    elif ice.__class__.__name__ == "spins":
        spins = spin_ice_vector(ice)
    elif ice.__class__.__name__ == "DataFrame":
        spins = trj_ice_vector(ice)
    elif ice.__class__.__name__ == "ndarray":
        spins = ice
    
    return spins 

def where_is_edge(e, edge_directory):
    """ What vertex in the edge directory contains the edge 'e'. """
    
    vertices = [i for i in edge_directory if np.isin(e, edge_directory[i])]
    
    if len(vertices)==1:
        vertices.append(-1)
    if len(vertices)!=2:
        print(vertices)
        raise ValueError("edges can only join two vertices")
        
    return vertices

def update_edge_directions(edges, spins, positions):
    """ Map the 'spins' to the edge directions in 'edges'. """


    for i,e in tqdm.tqdm(edges.iterrows(), len(edges)):

        spin_direction = spins["Direction"][e.name]

        if (e<0).any():
            # This happens when a single vertex is assigned to an edge 
            vertex = e[e>=0]
            if vertex.index[0]=="start":
                vertex_join = spins["Center"][e.name]-positions[vertex[0]]
            elif vertex.index[0]=="end":
                vertex_join = positions[vertex[0]]-spins["Center"][e.name]

        else:
            vertex_join = positions[e["end"]]-positions[e["start"]]

        if np.dot(spin_direction,vertex_join)<0:
            ## flip edge
            e[["start","end"]] = e[["end","start"]]

    return edges

def create_edge_array(edge_directory, spins = None, positions = None):
    """ Retrieve the edge array from the edge_directory. 
    If spins and positions are given they are used to calculate the directions of the edges. 
    """

    edge_ids = np.unique(np.array([e for v in edge_directory for e in tqdm.tqdm(edge_directory[v])]))
    
    edges = np.array([[e,*where_is_edge(e, edge_directory)] 
                      for e in progress_func(edge_ids)])
    
    edges = pd.DataFrame(data = edges[:,1:],
                         columns=["start","end"],
                         index=pd.Index(edges[:,0],name="edge"))
    
    if spins is not None and positions is not None:
        edges = update_edge_directions(edges,spins,positions)
    
    return edges
        
class vertices():
    def __init__(self, positions = None, edges = None, ice = None, id_label = "id", static = True, ):
        """ Initializes the vertices array.
        Initialization method for the vertices class. 
        Vertices are defined by a set of positions, and a set of directed edges. If any of these are given, then the processing is easier. If they are not given they are inferred from the `input`. If the input is not given, the vertex object is initialized empty, but a topology can be added later by using "colloids_to_vertices", "spins_to_vertices", or "trj_to_vertices". 
        If an object is given to create a vertex array, do so. 
        ---------
        Parameters:
        * positions: Ordered list containing the geometry of the vertices.
        * edges: Ordered list, or disordered set, containing the pairs of vertices that are joined. 
        * ice (colloidal_ice object, trj dataframe, spin_ice object): Initializes the topology, inferring from the input. 
            * colloidal_ice (colloidal_ice object, optional): Initalizes the vertices from a colloidal_ice object
            * spin_ice (spin_ice object, optional): Initializes the vertices from a spin_ice object
            * trj (pd.DataFrame, optional): Initializes the vertices from a pandas array. The pandas array must have the columns [x y z] and [dx dy dz] from which the links direction will be deduced. 
        * id_label (string, optional): If the index of `trj` has more than one level, this is the name of the level that identifies particles. Defaults to "id".
        * static (boolean, True): If the topology of the traps doesn't change, then time can be saved by not recalculating neighbors. Setting this variable to true indicates if static topology can be assumed in case of a MultiIndex. False is not implemented
        
        Attributes: 
        * vertices (DataFrame): contains the possitions of the vertices, plus whatever properties have been calculated. 
        * edges (DataFrame): contains the pairs of vertices that are connected. The edges are directed and go from the first vertex to the second vertex. 
        * edge_directory (dict): indicates which vertices are formed by which edges. The index is the edge number. Each entry contains a list of vertices.
        """
        
        self.vertices = pd.DataFrame({"x":[],"y":[]},
                            index = pd.Index([],name="vertex"))
        self.edges = pd.DataFrame({"start":[],"end":[]},
                             index = pd.Index([],name="edge"))
        self.edge_directory = {}
            
        if positions is not None:
            self.vertices.x = positions[:,0]
            self.vertices.y = positions[:,1]
            
        if edges is not None:
            self.edges.start = edges[:,0]
            self.edges.end = edges[:,1]
                        
    def infer_topology(self, ice, positions=None, method = "crossings", tolerance = 0.01):
        """ Infer the topology from the spin structure.
        ------------
        Parameters:
        input: object to get the spins from. 
        positions (optional): 
        method (string, "crossings"): Method to infer the positions of the vertices. 
            * "crossings" defines vertices as being in the crossing points of two spins. This is illdefined in more than 2D. 
            * "voronoi" defines vertices as being in the corners of the voronoi tesselation of  
        """
        spins = ice_to_spins(ice)

        neighbor_pairs = calculate_neighbor_pairs(spins['Center'])
        neighbor_pairs = from_neighbors_get_nearest_neighbors(neighbor_pairs)
        neighbor_pairs = get_vertices_positions(neighbor_pairs,spins)
        
        positions, inverse, copies = unique_points(neighbor_pairs['Vertex'])
        self.vertices.x = positions[:,0]
        self.vertices.y = positions[:,1]
        
        self.edge_directory = {i:np.unique(neighbor_pairs[c]["Pair"].flatten()) 
                                for i,c in enumerate(copies)}

        self.edges = create_edge_array(self.edge_directory, spins, positions)   
    
    def update_directions(self, ice):
        """ Updates the directions of the vertices using an ice object """
        
        positions = self.vertices.loc[:,["x","y"]].values
        spins = ice_to_spins(ice)
        
        self.edges = update_edge_directions(self.edges, spins, positions)
        
    def calculate_coordination(self):
        """ Adds a column to the 'vertices' array with the vertex coordination """
        coordination = [len(self.edge_directory[vertex]) for vertex in self.vertices.index]
        self.vertices["coordination"] = coordination
         
    def calculate_charge(self):
        """ Adds a column to the 'vertices' array with the vertex charge. """
        
        self.vertices["charge"] = 0

        for v_id, vertex in self.vertices.iterrows():
            indegree = (self.edges.loc[self.edge_directory[v_id]].end==v_id).sum()
            outdegree = (self.edges.loc[self.edge_directory[v_id]].start==v_id).sum()
            self.vertices.loc[v_id,"charge"] = indegree-outdegree
    
    def calculate_dipole(self, spins):
        """ Adds two column sto the 'vertices' array with the sum of the directions of the vertex components. """
    
        self.vertices["dx"] = 0
        self.vertices["dy"] = 0

        for v_id, vertex in self.vertices.iterrows():
            self.vertices.loc[v_id,["dx","dy"]] = np.sum(np.array(
                        [spins["Direction"][e] for e in self.edge_directory[v_id]]),
                    axis=0)
            
    def classify_vertices(self, spins):
        
        self.calculate_coordination()
        self.calculate_charge()
        self.calculate_dipole(spins)
         
        return self

    def colloids_to_vertices(self, col):
        """ Uses the col object to infer the topology of the vertices and to classify them."""
        
        spins = ice_to_spins(col)
        self.infer_topology(spins)
        self.classify_vertices(spins)
        
        return self
    
    def trj_to_vertices(self, trj, positions = None, id_label = None, static = True):
        """ Convert a trj into a vertex array. 
        If trj is a MultiIndex, an array will be saved that has the same internal structure as the passed array, but the identifying column will now refer to vertex numbers. 
        ---------
        Parameters: 
        * trj (pd.DataFrame, optional): Initializes the vertices from a pandas array. The pandas array must have the columns [x y z] and [dx dy dz] from which the links direction will be deduced. 
        * id_label (string, "id"): If the index of `trj` has more than one level, this is the name of the level that identifies particles.
        * static (boolean, True): If the topology of the traps doesn't change, then time can be saved by not recalculating neighbors. Setting this variable to true indicates if static topology can be assumed in case of a MultiIndex.
        """
                
        def trj_to_vertices_single_frame(trj_frame):
            
            spins = ice_to_spins(trj_frame)
            
            if len(self.vertices)==0:
                self.infer_topology(spins, positions=positions)
            else: 
                self.update_directions(spins) 
                
            self.classify_vertices(spins)
            
            return self.vertices
            
        if trj.index.nlevels==1:
                        
            trj_to_vertices_single_frame(trj)
            
            return self
            
        else:
                
            id_i = np.where([n=="id" for n in trj.index.names])
            other_i = list(trj.index.names)
            other_i.remove(other_i[id_i[0][0]])
            
            self.dynamic_array = trj.groupby(other_i).apply(trj_to_vertices_single_frame)
            self.vertices = self.dynamic_array
            
            return self
    
    def display(self, ax = None, DspCoord = False, dpl_scale = 1, dpl_width = 5, sl=None):
        
        
        if self.vertices.index.nlevels>1:
            if sl is None:
                sl = self.vertices.index[-1][:-1]
            sl = sl+(slice(None),)
        else: 
            sl = slice(None)
        
        vertices = self.vertices.loc[sl]

        if ax is None:
            ax = plt.gca()

        if not DspCoord:
            for i,v in vertices.iterrows():
                if v.charge>0:
                    c = 'r'
                else:
                    c = 'b'
                ax.add_patch(patches.Circle((v.x,v.y),radius = abs(v['charge'])*2,
                    ec='none', fc=c))

                if v.charge==0:
                    X = v.x
                    Y = v.y
                    
                    DX = v['dx']*dpl_scale
                    DY = v['dy']*dpl_scale
                    ax.add_patch(patches.Arrow(X-DX,Y-DY,2*DX,2*DY,width=dpl_width,fc='k'))
                
        if DspCoord: 
            for v in vertices.iterrows:
                if v['charge']>0:
                    c = 'r'
                else:
                    c = 'b'
                    
                ax.add_patch(patches.Circle((v.x,v.y),radius = abs(v['charge'])*2,
                    ec='none', fc=c))
                    
                X = v.x
                Y = v.y
        
        #ax.set_aspect("equal")    
        #plt.axis("equal")

#### Graph Class Definition ####
class graph():
    def __init__(self):

        self.edges = []
        self.vertices = []
        self.edge_directory = {}

    def __str__(self):
        return "Graph object with %u vertices and %u edges"%(len(self.vertices),len(self.edges))

    def display(self, ax = None, decimation = None):
        if ax is None:
            ax = plt.gca()

        ax.plot(self.vertices.x,self.vertices.y,'o')
        
        if decimation is not None:
            ax.plot(self.vertices.loc[decimation.values.flatten()].x,
                        self.vertices.loc[decimation.values.flatten()].y, '.')
            

        centers, directions = self.spins()

        v_points  = np.concatenate(
        [ centers-directions/2,
        centers+directions/2 ],axis=1)

        ax.plot(v_points[:,[0,3]].transpose(),v_points[:,[1,4]].transpose(),color="k")

        ax.set_aspect("equal")

    def spins_to_graph(self, spins, periodic = False, region = None):
        """returns a database of spins, a database of vertices and member directories both ways"""

        # Create a database of spins
        centers = np.array([s.center.magnitude for s in spins])
        directions = np.array([s.direction.magnitude for s in spins])

        sp_array = pd.concat([
            pd.DataFrame(data = centers, columns = ["x","y","z"]),
            pd.DataFrame(data = directions, columns = ["dx","dy","dz"])], axis = 1)
        sp_array.index.name = "id"
        sp_array.head()

        # Create two vertices for each spin
        vert = pd.concat([
            sp_array[["x","y","z"]]+sp_array[["dx","dy","dz"]].values/2,
            sp_array[["x","y","z"]]-sp_array[["dx","dy","dz"]].values/2])
        vert["sp_id"] = vert.index
        vert.index = range(len(vert))
        vert.index.name = "v_id"

        # Generate a unique array of vertices
        vert = np.round(vert*1e6)*1e-6

        if periodic:
            vert[["x","y","z"]] = np.mod(vert[["x","y","z"]],region)
            vert = np.round(vert*1e6)*1e-6

            self.periodic=True
            self.region = region

        v, ind, invind = np.unique(vert[["x","y","z"]].values, axis = 0, return_index = True, return_inverse = True)
        vert_un = pd.DataFrame(data = v, columns = ["x","y","z"], index = range(len(ind)))
        vert_un.index.name = "v_id"
        vert_un = vert_un.sort_index()

        # Directory of edges that point to or from a vertex.
        sp_id = {}

        for i,v in vert_un.iterrows():
            sp_id[i] = list(vert.loc[np.where(invind==i)[0]].sp_id.values)

        # Directory of vertices that edges point to.

        v_id = {}
        for i,sp in sp_array.iterrows():
            v_id[i] = invind[vert[vert.sp_id==i].index]

        self.vertices = vert_un
        self.edges = pd.DataFrame(data = v_id, index=["v_1","v_2"]).transpose()
        self.edges.index.name="e_id"

        self.edge_directory = sp_id

        return self

    def decimation(self,rm_edges):
        """ Remove an edge """
        self.latest = "decimation"
        self.edge_dec = self.edges.loc[[rand.choice(self.edges.index)]]

        edge = self.edge_dec
        self.edges = self.edges.drop(edge.index)

        self.edge_directory[edge["v_1"].values[0]].remove(edge.index)
        self.edge_directory[edge["v_2"].values[0]].remove(edge.index)

        rm_edges = rm_edges.append(edge)

        return rm_edges
    
    def undo_decimation(self,rm_edges):
        """ Replace the last edge that was removed (it shoud be the last in the list)"""
        
        if len(rm_edges)>0:
        
            edge = self.edge_dec
            self.edge_dec = None
            self.edges = self.edges.append(edge)

            self.edge_directory[edge["v_1"].values[0]].append(edge.index.values[0])
            self.edge_directory[edge["v_2"].values[0]].append(edge.index.values[0])

            rm_edges = rm_edges.drop(edge.index)
            
        return rm_edges
        
    def permutation(self,rm_edges):
        """ Remove a random edge and place it elsewhere"""
        self.latest = "permutation"

        if len(rm_edges)>0:

            self.edge_permute = [self.edges.loc[[rand.choice(self.edges.index)]], rm_edges.loc[[rand.choice(rm_edges.index)]]]
            edge_out = self.edge_permute[0]
            edge_in = self.edge_permute[1]

            self.edges = self.edges.append(edge_in)
            self.edges = self.edges.drop(edge_out.index)

            self.edge_directory[edge_out["v_1"].values[0]].remove(edge_out.index)
            self.edge_directory[edge_out["v_2"].values[0]].remove(edge_out.index)

            self.edge_directory[edge_in["v_1"].values[0]].append(edge_in.index.values[0])
            self.edge_directory[edge_in["v_2"].values[0]].append(edge_in.index.values[0])

            rm_edges = rm_edges.drop(edge_in.index)
            rm_edges = rm_edges.append(edge_out)

        return rm_edges

    def undo_permutation(self,rm_edges):
        """ undo the latest permutation"""

        if len(rm_edges)>0:

            edge_out = self.edge_permute[1]
            edge_in = self.edge_permute[0]
            
            self.edge_permute = None

            self.edges = self.edges.append(edge_in)
            self.edges = self.edges.drop(edge_out.index)

            self.edge_directory[edge_out["v_1"].values[0]].remove(edge_out.index)
            self.edge_directory[edge_out["v_2"].values[0]].remove(edge_out.index)

            self.edge_directory[edge_in["v_1"].values[0]].append(edge_in.index.values[0])
            self.edge_directory[edge_in["v_2"].values[0]].append(edge_in.index.values[0])

            rm_edges = rm_edges.drop(edge_in.index)
            rm_edges = rm_edges.append(edge_out)

        return rm_edges
        
    def refilling(self,rm_edges):
        """ Replace an edge from the list rm_edges"""
        self.latest = "refilling"
        
        if len(rm_edges)>0:
            
            self.edge_refill = rm_edges.loc[[rand.choice(rm_edges.index)]]
            edge = self.edge_refill
            self.edges = self.edges.append(edge)

            self.edge_directory[edge["v_1"].values[0]].append(edge.index.values[0])
            self.edge_directory[edge["v_2"].values[0]].append(edge.index.values[0])

            rm_edges = rm_edges.drop(edge.index)

        return rm_edges
    
    def undo_refilling(self, rm_edges):
        """ remove the last link which was replaced """
        
        edge = self.edge_refill
        self.edge_refill = None
        
        self.edges = self.edges.drop(edge.index)

        self.edge_directory[edge["v_1"].values[0]].remove(edge.index)
        self.edge_directory[edge["v_2"].values[0]].remove(edge.index)

        rm_edges = rm_edges.append(edge)

        return rm_edges
        
    def undo(self, rm_edges):
        if self.latest == "decimation":
            return self.undo_decimation(rm_edges)
        if self.latest == "permutation":
            return self.undo_permutation(rm_edges)
        if self.latest == "refilling":
            return self.undo_refilling(rm_edges)
            
    def copy(self, deep=True):
        if deep:
            return cp.deepcopy(self)
        else:
            return cp.copy(self)

    def spins(self, units = 1):

        centers = (self.vertices.loc[self.edges.v_1,["x","y","z"]].values+self.vertices.loc[self.edges.v_2,["x","y","z"]].values)/2
        directions = (self.vertices.loc[self.edges.v_1,["x","y","z"]].values-self.vertices.loc[self.edges.v_2,["x","y","z"]].values)

        if self.periodic:

            fwloop = directions-self.region
            bwloop = directions+self.region

            cnreg = self.region
            cnreg[np.isinf(cnreg)] = 0
            ctloop = centers + cnreg/2
            centers[abs(directions)>self.region/2] = ctloop[abs(directions)>self.region/2]
            centers[directions<-self.region/2] = ctloop[directions<-self.region/2]
            directions[directions>self.region/2] = fwloop[directions>self.region/2]
            directions[directions<-self.region/2] = bwloop[directions<-self.region/2]

        return centers*units, directions*units
