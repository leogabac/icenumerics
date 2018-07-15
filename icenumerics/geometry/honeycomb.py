import numpy as np
import random
import scipy.spatial as spa
    
def honeycomb_spin_ice_geometry(Sx,Sy,lattice,border):
    """This function calculates the positions and directions of the spins in a honeycomb spin ice system. 

    These are the arrays to iterate. For now, each point in x-y generates one unit cell which is a hexagon of spins. Then repeated spins are eliminated."""
        
    x = np.arange(0,Sx)
    y = np.arange(0,Sy)
    
    
    if border == "closed spin":
        t = np.arange(0,2*np.pi,np.pi/3)
        unit_cell = np.array([
            np.cos(t),
            np.sin(t),
            np.zeros(len(t)),
            -np.sin(t)/np.tan(np.pi/3),
            np.cos(t)/np.tan(np.pi/3),
            np.zeros(len(t))
            ])
    elif border == "closed vertex":
        t = (np.array([60,120,240,300]))/180*np.pi
        unit_cell = np.array([
            np.append([0],np.cos(t-np.pi/3)),
            np.append([0],np.sin(t-np.pi/3)),
            np.zeros(len(t)+1),
            np.append([np.sin(np.pi/3)],np.sin(t+np.pi/3))/np.tan(np.pi/3),
            np.append([np.cos(np.pi/3)],np.cos(t+np.pi/3))/np.tan(np.pi/3),
            np.zeros(len(t)+1)
            ])
    elif border == "periodic":
        t = np.arange(np.pi,2*np.pi,np.pi/3)
        unit_cell = np.array([
            1+np.cos(t),
            2/np.tan(np.pi/3)+np.sin(t),
            np.zeros(len(t)),
            -np.sin(t)/np.tan(np.pi/3),
            np.cos(t)/np.tan(np.pi/3),
            np.zeros(len(t))
            ])
    else: 
        raise(ValueError(border+" is not a supporteed border type."))
        
    lattice_X = np.mod(x+y[:,np.newaxis]*np.cos(np.pi/3),Sx).flatten()
    lattice_Y = (np.zeros(x.shape)+y[:,np.newaxis]*np.sin(np.pi/3)).flatten()

    centers = np.array([
        (lattice_X+1/2*unit_cell[0,:].reshape(len(unit_cell[0,:]),1)).flatten(),
        (lattice_Y+1/2*unit_cell[1,:].reshape(len(unit_cell[0,:]),1)).flatten(),
        (0*lattice_Y+unit_cell[2,:].reshape(len(unit_cell[0,:]),1)).flatten()]
        ).transpose()

    directions = np.array([
        (0*lattice_X+1*unit_cell[3,:].reshape(len(unit_cell[0,:]),1)).flatten(),
        (0*lattice_Y+1*unit_cell[4,:].reshape(len(unit_cell[0,:]),1)).flatten(),                 (0*lattice_Y+0*unit_cell[5,:].reshape(len(unit_cell[0,:]),1)).flatten(),
        ]).transpose()

    """This erases repeated spins"""
    """
    For this we find all neighbors within a small tolerance (using cKDTree is fast).
    We then make an array of all ids to remove by listing only the second member of each neighbor pair.
    We make a mask with the remove array and apply it to the arrays Center and Direction """
    tree = spa.cKDTree(centers)
    remove = [p[1] for p in tree.query_pairs(1e-10)]
    
    mask = np.ones(len(centers),dtype=bool)
    mask[remove] = False
    
    centers = centers[mask]
    directions = directions[mask]
    
    if border == "periodic":
        centers[:,0] = np.mod(centers[:,0],Sx)
        
    centers = centers*lattice
    directions = directions*lattice
    return centers, directions

