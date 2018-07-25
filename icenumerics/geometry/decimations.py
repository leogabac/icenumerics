import numpy as np
import random
import copy as cp 
from . import *
import scipy.spatial as spa 

def triangle_to_honeycomb(decimate_angles,centers,directions,lattice):
    
    units = lattice.units
    
    centers = centers/lattice
    directions = directions/lattice
    
    def intersect_with_tol(A,B,tol=1e-5):
        """ 
        Intersects two arrays within a tolerance. 
        Returns a logic array of size len(A), which indicates which points in A are present in B 
        """
        A_tree = spa.KDTree(A)
        B_tree = spa.KDTree(B)

        A_points_in_B = B_tree.query_ball_tree(A_tree,tol)
        A_points_in_B = [item for sublist in A_points_in_B for item in sublist]

        is_in_B = np.array([False]*len(centers))
        is_in_B[A_points_in_B] = True
        
        return is_in_B
    
    def sublattice_to_keep(centers):
    
        size = np.max(centers,0)-np.min(centers,0)
        lattice_size = [np.ceil(size[1]/2/np.cos(np.pi/6))+1,np.ceil(size[0]/2/np.cos(np.pi/6)**2)]
    
        centers_hex,dir_hex = honeycomb_spin_ice_geometry(
            lattice_size[0],lattice_size[1],
            1*units,border="closed spin"
            )
        centers_hex = centers_hex/units
        centers_hex[:,0:2] = centers_hex[:,1::-1]
        centers_hex = centers_hex[:,:]*np.tan(np.pi/3)+[1,0.5*np.tan(np.pi/3),0]
    
        return centers_hex
        
    def vertex_direction_sublattices(centers):

        size = np.max(centers,0)-np.min(centers,0)
    
        lattice_size = [np.ceil(size[1]/2/np.cos(np.pi/6)),np.ceil(size[0]/2/np.cos(np.pi/6)**2)]
        #sp_tri = ice.spins()
        #sp_tri.create_lattice("triangular",
                              #lattice_size,
                              #lattice_constant=1*ureg.um,border = "closed spin")
        centers_tri_a,dir_tri = triangular_spin_ice_geometry(
            lattice_size[0],lattice_size[1],1*units,border="closed spin"
            )
        #centers_tri_a = np.array([s.center/sp_tri.lattice for s in sp_tri])
        centers_tri_a = centers_tri_a/units
        centers_tri_a[:,0:2] = centers_tri_a[:,1::-1]
        centers_tri_a = centers_tri_a*np.tan(np.pi/3)-[0,0,0]

        centers_tri_b = centers_tri_a+[-1,0,0]
    
        return centers_tri_a, centers_tri_b
    
    def angle_array(centers,directions):

        vertex_sub_a, vertex_sub_b = vertex_direction_sublattices(centers)

        sublattice_a = intersect_with_tol(centers,vertex_sub_a)
        sublattice_b = intersect_with_tol(centers,vertex_sub_b)

        angles = np.round(np.mod(np.arctan2(directions[:,1],directions[:,0]),np.pi)/(np.pi)*3)

        angles[sublattice_b] = np.mod(3-angles[sublattice_b]*2,6)
        angles[sublattice_a] = np.mod(-angles[sublattice_a]*2,6)
    
        return angles

    keep_sublattice = sublattice_to_keep(centers)
    decimate = intersect_with_tol(centers,keep_sublattice)==False

    angles = angle_array(centers,directions)
    angles[np.array(decimate)==False] = np.NaN
    
    return [a not in np.mod(decimate_angles,6) for a in angles]