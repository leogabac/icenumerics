import numpy as np
import scipy.spatial as spa

def SquareSpinIceConcatenateCenters(VSpinsX,VSpinsY,HSpinsX,HSpinsY,Lattice,Boundary):
    """ Concatenate Vertical and Horizontal spin arrays """
    if Boundary == "closed spin":
        Delta = 0.5
    elif Boundary == "closed vertex":
        Delta = -0.5

    return np.vstack(
    (
        np.hstack(
            (
                VSpinsX, # X
                VSpinsY+Delta*Lattice, #Y 
                np.zeros(VSpinsX.shape,float) #Z
                ) # Centers of Vertical Spins
            ), 
         np.hstack(
            (
                HSpinsX+Delta*Lattice, #X
                HSpinsY, #Y
                np.zeros(HSpinsX.shape,float) #Z
                ) # Centers of Horizontal Spins
            )
        ) # (Centers Vertical, Centers Horizontal)
    ) # Centers of All Spins, Vertical stacked on top of Horizontal
    return Center

def SquareSpinIceDirection(VOrderDirectionArray,HOrderDirectionArray):
    return np.vstack(
    (
        np.hstack(
            (
                np.zeros(VOrderDirectionArray.shape,float),
                VOrderDirectionArray,
                np.zeros(VOrderDirectionArray.shape,float)
                )
            ),
         np.hstack(
            (
                HOrderDirectionArray,
                np.zeros(HOrderDirectionArray.shape,float),
                np.zeros(HOrderDirectionArray.shape,float)
                )
            )
        ) # (Directions Vertical, Directions Horizontal)
    ) # Direction of All Spins

def SquareSpinIceDirectionGSOrdering(VSpinsX,VSpinsY,HSpinsX,HSpinsY,Lattice):
    VOrderDirectionArray = np.ones(VSpinsX.shape,float)
    VOrderDirectionArray[
        np.mod(
            VSpinsX+VSpinsY,
            2*Lattice)>0]=-1

    HOrderDirectionArray = np.ones(HSpinsX.shape,float)
    HOrderDirectionArray[
        np.mod(
            HSpinsX+HSpinsY,
            2*Lattice)==0]=-1
    return VOrderDirectionArray, HOrderDirectionArray

def SquareSpinIceDirectionRandomOrdering(VSpinsX,VSpinsY,HSpinsX,HSpinsY,Lattice):
    VOrderDirectionArray = np.random.rand(*VSpinsX.shape)
    VOrderDirectionArray[VOrderDirectionArray>0.5]=1;
    VOrderDirectionArray[VOrderDirectionArray<=0.5]=-1;
    
    HOrderDirectionArray = np.random.rand(*HSpinsX.shape)
    HOrderDirectionArray[HOrderDirectionArray>0.5]=1;
    HOrderDirectionArray[HOrderDirectionArray<=0.5]=-1;

    return VOrderDirectionArray, HOrderDirectionArray

def square_spin_ice_geometry(sx,sy,lattice,border):
    """ 
    This function calculates the positions and directions of the spins in a square spin ice system. 
    
    It does so by creating two meshes, 
    where one mesh defines the position of the horizontal spins, 
    and the other defines the position of the vertical spins.
    Then a subroutine concatenates the arrays.
    """
    
    if border == "periodic":
    
        t = np.array([0,90])/180*np.pi
        unit_cell_center = lattice/2*np.array(
            [np.cos(t),np.sin(t),np.zeros(len(t))]).transpose()
        unit_cell_direction = lattice*np.array(
            [np.cos(t),np.sin(t),np.zeros(len(t))]).transpose()
   
    elif border == "closed spin":
        
        t = np.array([0,90])/180*np.pi
        
        unit_cell_center = np.array(
            [[lattice/2,0,0],
             [0,lattice/2,0],
             [lattice,lattice/2,0],
             [lattice/2,lattice,0]])
      
      
        
        unit_cell_direction = lattice*np.array(
            [[1,0,0],[0,1,0],[0,1,0],[1,0,0]])
    
    elif border == "closed vertex":
        
        t = np.array([0,90])/180*np.pi
        
        unit_cell_center = np.array(
            [[lattice/2,0,0],
             [0,lattice/2,0],
             [lattice,lattice/2,0],
             [lattice/2,lattice,0]])
      
      
        
        unit_cell_direction = lattice*np.array(
            [[0,-1,0],[1,0,0],[1,0,0],[0,1,0]])
        
        
    else: 
        raise(ValueError(border+" is not a supporteed border type."))
        
    space = np.meshgrid(
        np.arange(0,sx)*lattice,
        np.arange(0,sy)*lattice,
        np.arange(1))
        
    n = np.array([1,1,1])
    R = np.array([s.flatten()*n[i] for i,s in enumerate(space)])
    
    centers = np.concatenate([R.transpose()+c for c in unit_cell_center])
    directions = np.concatenate([np.zeros(np.shape(R.transpose()))+c for c in unit_cell_direction])
    
    tree = spa.cKDTree(centers)
    remove = [p[1] for p in tree.query_pairs(1e-10)]
    
    mask = np.ones(len(centers),dtype=bool)
    mask[remove] = False
    
    centers = centers[mask]
    directions = directions[mask]

    
    return centers, directions
    
def square_spin_ice_geometry3D(sx,sy,lattice,h,border):
    """ 
    This function calculates the positions and directions of the spins in a square spin ice system. 
    
    It does so by creating two meshes, 
    where one mesh defines the position of the horizontal spins, 
    and the other defines the position of the vertical spins.
    Then a subroutine concatenates the arrays.
    """
    
    if border == "periodic":
    
        t = np.array([0,90])/180*np.pi
        unit_cell_center = np.array(
            [np.cos(t)*lattice/2,np.sin(t)*lattice/2,np.sin(t)*h]).transpose()
        unit_cell_direction = lattice*np.array(
            [np.cos(t),np.sin(t),np.zeros(len(t))]).transpose()

    
    space = np.meshgrid(
        np.arange(0,sx)*lattice,
        np.arange(0,sy)*lattice,
        np.arange(1))
        
    n = np.array([1,1,1])
    R = np.array([s.flatten()*n[i] for i,s in enumerate(space)])
    
    centers = np.concatenate([R.transpose()+c for c in unit_cell_center])
    directions = np.concatenate([np.zeros(np.shape(R.transpose()))+c for c in unit_cell_direction])

    
    return centers, directions