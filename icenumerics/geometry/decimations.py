import numpy as np
import random
import scipy.spatial as spa
import copy as cp 
from . import *

def triangle_to_honeycomb(decimate_angles,centers,directions,lattice):
    
    units = lattice.units
    
    centers = centers/lattice
    directions = directions/lattice
    
    def ismember_tol(point,point_list,tol = 1e-5):
        """Determines if point is in the list point_list, within a tolerance tol"""
        for p in point_list:
            if np.all(np.isclose(p,point,atol=tol)):
                return True
        return False
        
    def sublattice_to_keep(centers):
    
        size = np.max(centers,0)-np.min(centers,0)
        lattice_size = [np.ceil(size[1]),np.ceil(size[0])]
    
#        sp_hex = ice.spins()
#        sp_hex.create_lattice("honeycomb",lattice_size,lattice_constant=1*ureg.um,border = "closed spin")
        centers_hex,dir_hex = honeycomb_spin_ice_geometry(
            lattice_size[0],lattice_size[1],
            1*units,border="closed spin"
            )
 #       centers_hex = np.array([s.center/sp_hex.lattice for s in sp_hex])
        centers_hex = centers_hex/units
        centers_hex = centers_hex[:,::-1]*np.tan(np.pi/3)+[1,0.5*np.tan(np.pi/3)]
    
        return centers_hex
        
    def vertex_direction_sublattices(centers):

        size = np.max(centers,0)-np.min(centers,0)
    
        lattice_size = np.ceil(size/np.tan(np.pi/3))[::-1]+2
        #sp_tri = ice.spins()
        #sp_tri.create_lattice("triangular",
                              #lattice_size,
                              #lattice_constant=1*ureg.um,border = "closed spin")
        centers_tri_a,dir_tri = triangular_spin_ice_geometry(
            lattice_size[0],lattice_size[1],1*units,border="closed spin"
            )
        #centers_tri_a = np.array([s.center/sp_tri.lattice for s in sp_tri])
        centers_tri_a = centers_tri_a/units
        centers_tri_a = centers_tri_a[:,::-1]*np.tan(np.pi/3)-[0,0]

        centers_tri_b = centers_tri_a+[-1,0]
    
        return centers_tri_a, centers_tri_b
    
    def angle_array(centers,directions):

        vertex_sub_a, vertex_sub_b = vertex_direction_sublattices(centers)

        sublattice_a = [ismember_tol(c_tri,vertex_sub_a) for c_tri in centers]
        sublattice_b = [ismember_tol(c_tri,vertex_sub_b) for c_tri in centers]

        angles = np.mod(np.arctan2(directions[:,1],directions[:,0]),np.pi)/(np.pi)*3

        angles[sublattice_b] = np.mod(3-angles[sublattice_b]*2,6)
        angles[sublattice_a] = np.mod(-angles[sublattice_a]*2,6)
    
        return angles

    keep_sublattice = sublattice_to_keep(centers)
    decimate = [not ismember_tol(c_tri,keep_sublattice) for c_tri in centers]
    angles = angle_array(centers,directions)
    angles[decimate] = np.NaN
    print(angles)
    
    return [a not in decimate_angles for a in angles]