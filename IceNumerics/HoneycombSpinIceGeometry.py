import numpy as np
import random

def HoneycombSpinIceDirectionRandomOrdering(Direction):

    Direction = np.array([dir*random.randrange(-1,2,2) for dir in Direction])
    return Direction

def HoneycombSpinIceCalculateGeometry(Sx,Sy,Lattice,Ordering):
    # This function calculates the positions and directions of the spins
    # in a honeycomb spin ice system. 

    # These are the arrays to iterate. For now, each point in x-y generates one
    # unit cell which is a hexagon of spins. Then repeated spins are eliminated.
    x = np.arange(0,Sx) * Lattice
    y = np.arange(0,Sy) * Lattice
    t = np.arange(0,2*np.pi,np.pi/3)

    CenterX = np.mod(x+y[:,np.newaxis]*np.cos(np.pi/3),Sx*Lattice) + \
        Lattice*np.cos(t.reshape(t.size,1,1))*0.5
    CenterY = np.zeros(x.shape)+y[:,np.newaxis]*np.sin(np.pi/3) + \
        Lattice*np.sin(t.reshape(t.size,1,1))*0.5

    Center = np.vstack([CenterX.flatten(), \
        CenterY.flatten(), \
        np.zeros(CenterX.flatten().shape)]).T

    DirectionX = np.zeros(x.shape) + np.zeros(y[:,np.newaxis].shape) + \
        (np.cos(t.reshape(t.size,1,1)))
    
    DirectionY = np.zeros(x.shape) + np.zeros(y[:,np.newaxis].shape) - \
        (np.sin(t.reshape(t.size,1,1)))

    Direction = np.vstack([DirectionY.flatten(), \
        DirectionX.flatten(), \
        np.zeros(DirectionX.flatten().shape)]).T

    # This erases repeated spins
    for i in range(len(Center)):
        for j in range(i):
            if np.allclose(Center[i],Center[j],rtol=1e-10):
                Center[j] = np.NaN
                Direction[j] = np.NaN

    Center = Center[~np.isnan(Center[:,0]),:]
    Direction = Direction[~np.isnan(Direction[:,0]),:]
        
    if Ordering == "Random":
        Direction = HoneycombSpinIceDirectionRandomOrdering(Direction)

    elif Ordering == "GroundState":
        error("Sorry, I haven't programed this yet")
    elif Ordering == "Biased":
        error("Sorry, I haven't programed this yet")
    else:
        error("I do not know this ordering")
    
#    for c in Center:
#        c = RoundToN(c,6)
    
    return Center, Direction

#def RoundToN(X,n):
#    X = np.array([np.around(x,n-np.int(np.floor(np.log10(np.abs(x))))) if x!=0 else 0 for x in X])
#    return X
