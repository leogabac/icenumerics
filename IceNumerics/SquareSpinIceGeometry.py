import numpy as np

def SquareSpinIceCenters(VSpinsX,VSpinsY,HSpinsX,HSpinsY,Lattice):
    return np.vstack(
    (
        np.hstack(
            (
                VSpinsX, # X
                VSpinsY+0.5*Lattice, #Y 
                np.zeros(VSpinsX.shape,float) #Z
                ) # Centers of Vertical Spins
            ), 
         np.hstack(
            (
                HSpinsX+0.5*Lattice, #X
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

def SquareSpinIceCalculateGeometry(Sx,Sy,Lattice,Ordering,Ratio):
    # This function calculates the positions and directions of the spins
    # in a square spin ice system. 
    
    # Create two meshes,
    # One defines the positions of horizontal spins,
    # The other defines the positions of vertical spins.
    # These meshes are then flatened and turned to vertical arrays so that
    # they can be concatenated to form the array of centers. 
    
    vsoX = np.arange(0,Sx+1)*Lattice
    vsoY = np.arange(0,Sy+1)*Lattice

    VSpinsX, VSpinsY = np.meshgrid(vsoX,vsoY[0:-1])
    HSpinsX, HSpinsY = np.meshgrid(vsoX[0:-1],vsoY)

    VSpinsX = VSpinsX.flatten(1)[None].T
    VSpinsY = VSpinsY.flatten(1)[None].T
    HSpinsX = HSpinsX.flatten(1)[None].T
    HSpinsY = HSpinsY.flatten(1)[None].T

    Center = SquareSpinIceCenters(VSpinsX,VSpinsY,HSpinsX,HSpinsY,Lattice)

    if Ordering == "Random":
        VOrderDirectionArray, HOrderDirectionArray = \
            SquareSpinIceDirectionRandomOrdering(
                VSpinsX,VSpinsY,HSpinsX,HSpinsY,Lattice)

    elif Ordering == "GroundState":
        
        VOrderDirectionArray, HOrderDirectionArray = \
            SquareSpinIceDirectionGSOrdering(
                VSpinsX,VSpinsY,HSpinsX,HSpinsY,Lattice)
            
    elif Ordering == "Biased":
        VOrderDirectionArray = np.ones(VSpinsX.shape,float)
        HOrderDirectionArray = np.ones(HSpinsX.shape,float)

    else:
        error("I do not know this ordering")

    Direction = SquareSpinIceDirection(VOrderDirectionArray*Ratio,HOrderDirectionArray)

    return Center, Direction
