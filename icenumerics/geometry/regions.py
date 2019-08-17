import numpy as np
import random
import scipy.spatial as spa

def triangular_region_periodic(spins):
    
    units = spins[0].center.units
    
    starts = np.array([(s.center + s.direction/2).to(units).magnitude 
            for s in spins])
    ends = np.array([(s.center - s.direction/2).to(units).magnitude 
            for s in spins])
    
    vertices = np.concatenate([starts,ends])
            
    lower_bounds = np.min(vertices,axis=0)
    
    vertices = vertices[vertices[:,0]>(lower_bounds[0]+1e-5)]
    lower_bounds = np.min(vertices,axis=0)
    upper_bounds = np.max(vertices,axis=0)
    
    return np.vstack([lower_bounds,upper_bounds])*units
    

#### Below this is probably trash. 

def calculate_neighbor_pairs(centers):
    """This function makes a list of all the Pairs of Delaunay Neighbors from an array of points"""
    
    tri = spa.Delaunay(centers)

    # List all Delaunay neighbors in the system
    neighbor_pairs = np.array(np.zeros(2*np.shape(tri.simplices)[0]),
                dtype=[('Pair',np.int,(2,)),('Distance',np.float),('Vertex',np.float,(2,))])

    i = 0
    for t in tri.simplices:
        neighbor_pairs[i]['Pair'] = np.sort(t[0:2])
        neighbor_pairs[i]['Distance'] = spa.distance.euclidean(centers[t[0]],centers[t[1]])
        neighbor_pairs[i+1]['Pair'] = np.sort(t[1:3])
        neighbor_pairs[i+1]['Distance'] = spa.distance.euclidean(centers[t[1]],centers[t[2]])
        i = i+2

    return neighbor_pairs
    
def from_neighbors_get_nearest_neighbors(neighbor_pairs):
    """ This function takes a list of Delaunay Neighbor Pairs and returns only those which are close to the minimum distance. """
    neighbor_pairs['Distance']=np.around(neighbor_pairs['Distance'],decimals=4)
    neighbor_pairs = neighbor_pairs[neighbor_pairs['Distance']<=np.min(neighbor_pairs['Distance'])*1.1]
    
    return neighbor_pairs
    
    
def get_vertices_positions(neighbor_pairs,centers,directions):
    # From a list of Spins, get neighboring spins, and get the crossing point of each, which defines a vertex.  
    for i,n in enumerate(neighbor_pairs):
        neighbor_pairs[i]['Vertex'] = spin_crossing_point(centers[n['Pair']][:,:2],directions[n['Pair']][:,:2])[0:2]
    
    return neighbor_pairs
    
def spin_crossing_point(centers,directions):
    # This works well in 2d. In 3d it's triciker
    if not (directions[0]==directions[1]).all():
        A = directions.transpose()
        A[:,1] = -A[:,1]

        b = np.array(np.diff(centers,axis=0)).transpose()

        lam = np.linalg.solve(A,b)

        return  centers[0,:]+lam[0]*directions[0,:]
    else:
        return np.Inf+np.zeros(centers)
        
def unique_points(Points,Tol = 0.1):
    """Returns only the distinct points (with a tolerance)."""
    Distance = spa.distance.squareform(spa.distance.pdist(Points))
    IsLast = []
    for i,p in enumerate(Points):
        IsLast = IsLast + [not np.any(Distance[(i+1):,i]<=Tol)]
        
    return Points[np.array(IsLast),:]
            
def calculate_periodic_region(centers,directions):
    
    neighbor_pairs = calculate_neighbor_pairs(centers)
    neighbor_pairs = from_neighbors_get_nearest_neighbors(neighbor_pairs)
    neighbor_pairs = get_vertices_positions(neighbor_pairs,centers,directions)
    return unique_points(neighbor_pairs['Vertex'])