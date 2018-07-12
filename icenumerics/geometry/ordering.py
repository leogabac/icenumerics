import numpy as np
import random
import scipy.spatial as spa

def random_ordering(centers,lattice):

    directions = np.array([random.randrange(-1,2,2) for c in centers])
    return directions

def HoneycombSpinIceDirectionSpinSolidPhase(Direction,Center,Lattice):
    """ Honeycomb Spin Solid Phase """
    sqrt3 = np.sqrt(3)
    epsilon = 10**-6

    #To improve readibility, the quantities on this should be given names that are descriptive of their function. 
    
    # JJ1 are the vertical spins in even rows (starting from zero). One over 3 should all point down: those in 3*Lattice/2, 9*Lattice/2,  12*Lattice/2.
   
    JJ1= np.logical_or(
        (Center[:,1]/Lattice) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice)% (sqrt3) - sqrt3) < epsilon)

    Direction[JJ1]=abs(Direction[JJ1])

    JJ12 = np.logical_and(JJ1,np.logical_or
                           ((Center[:,0]/Lattice -3/2) % 3/2 < epsilon, 
                            abs((Center[:,0]/Lattice -3/2) % 3/2 -3/2)<epsilon)) 

    Direction[JJ12]=-Direction[JJ12]

    # JJ2 are the tilted spins below the even rows. The 3 first (starting from 0) should point right, the 3 following should point left, ect..

    JJ2 = np.logical_or(
        ((Center[:,1]/Lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
        abs((Center[:,1]/Lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon)

    # JJ5 are the spins in JJ2 that should point right.

    JJ5=np.logical_and(JJ2,(((((4*Center[:,0])/Lattice)-1 + epsilon)//2)//3 % 2) < epsilon)
        

    JJ7=np.logical_and(JJ5,Direction[:,0]>0)

    Direction[JJ7]=-Direction[JJ7]

    # JJ9 are the spins in JJ2 that should point left.


    JJ9=np.logical_and(JJ2,np.logical_not((((((4*Center[:,0])/Lattice)-1 + epsilon)//2)//3 % 2) < epsilon))

    JJ11=np.logical_and(JJ9,Direction[:,0]<0)

    Direction[JJ11]=-Direction[JJ11]

    # JJ4 are vertical spins in odd rows. One over 3 should all point down: those in 0, 3*Lattice,  6*Lattice.

    JJ4 = np.logical_or(
        ((Center[:,1]/Lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon)

    Direction[JJ4]=abs(Direction[JJ4])

    JJ42 = np.logical_and(JJ4, (Center[:,0]/Lattice) % 3 < epsilon)

    Direction[JJ42]=-Direction[JJ42]

    # JJ3 are the tilted spins above the even rows. 
    JJ3 = np.logical_or(
        ((Center[:,1]/Lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon)

    JJ6=np.logical_and(JJ3,(((((4*Center[:,0])/Lattice)-1 + epsilon)//2)//3 % 2) < epsilon)

    JJ8=np.logical_and(JJ3,Direction[:,0]<0)

    Direction[JJ8]=-Direction[JJ8]

    JJ10=np.logical_and(JJ3,np.logical_not(((((((4*Center[:,0])/Lattice)-1 + epsilon)//2)//3 % 2) < epsilon)))
     
    JJ12=np.logical_and(JJ10,Direction[:,0]>0)

    Direction[JJ12]=-Direction[JJ12]

    return Direction

def HoneycombSpinIceDirectionBandAntiferromagnetic(Direction,Center,Lattice):
    """This state has lines in one direction in a ferromagnetic state with alternating orientation, and bands in the other directions in an antiferromagnetic state."""
    sqrt3 = np.sqrt(3)
    epsilon = 10**-6
    
    # JJ1 are the vertical spins in even rows (starting from zero). These should all point up. 
    JJ1= np.logical_or(
        (Center[:,1]/Lattice) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice)% (sqrt3) - sqrt3) < epsilon)


    Direction[JJ1,1]=abs(Direction[JJ1,1])

   # JJ2 are the tilted spins below the even rows. These should all point right
    JJ2 = np.logical_or(
        ((Center[:,1]/Lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
        abs((Center[:,1]/Lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon)
    
    # JJ5 are the spins in JJ2 that point left. These we flip.
    JJ5 = np.logical_and(JJ2, Direction[:,0]<0)
    Direction[JJ5] = -Direction[JJ5]
    
    # JJ3 are the tilted spins above the even rows. These should all point left
    JJ3 = np.logical_or(
        ((Center[:,1]/Lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon)
    
    # JJ7 are the spins in JJ3 that point right. These we flip
    JJ7= np.logical_and(JJ3,Direction[:,0]>0)
    Direction[JJ7]=-Direction[JJ7]

    # JJ4 are vertical spins in odd rows. These should all point down.
    JJ4 = np.logical_or(
        ((Center[:,1]/Lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon)

    Direction[JJ4]=-abs(Direction[JJ4])    
    
    return Direction

def HoneycombSpinIceDirectionBandMixedAntiferromagnetic(Center,Lattice,Direction):
    """The state has the bands in one direction in a ferromagnetic state, and the bands in the other two directions in an antiferromagnetic state."""
    
    sqrt3 = np.sqrt(3)
    epsilon = 10**-6

    JJ1= np.logical_and( 
         np.logical_or(
        (Center[:,1]/Lattice) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice)% (sqrt3) - sqrt3) < epsilon),
         np.logical_or(
        (Center[:,0]/Lattice -1/2) % (2) < epsilon,
        abs((Center[:,0]/Lattice -1/2))% (2) < epsilon))


    Direction[JJ1,1]=abs(Direction[JJ1,1])

    JJ11= np.logical_and( 
          np.logical_or(
         (Center[:,1]/Lattice) % (sqrt3) < epsilon,
         abs((Center[:,1]/Lattice)% (sqrt3) - sqrt3) < epsilon),
          np.logical_or(
         (Center[:,0]/Lattice +1/2) % (2) < epsilon,
         abs((Center[:,0]/Lattice +1/2))% (2) < epsilon))

    Direction[JJ11,1]=-abs(Direction[JJ11,1])

     # JJ2 are the tilted spins below the even rows. These should all point right
    JJ2 = np.logical_or(
        ((Center[:,1]/Lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
        abs((Center[:,1]/Lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon)
    
    # JJ5 are the spins in JJ2 that point left. These we flip.
    JJ5 = np.logical_and(JJ2, Direction[:,0]<0)
    Direction[JJ5] = -Direction[JJ5]
    
    # JJ3 are the tilted spins above the even rows. These should all point left
    JJ3 = np.logical_or(
        ((Center[:,1]/Lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon)
    
    # JJ7 are the spins in JJ3 that point right. These we flip
    JJ7= np.logical_and(JJ3,Direction[:,0]>0)
    Direction[JJ7]=-Direction[JJ7]

   # JJ4 are vertical spins in odd rows. These should all point down.
    JJ4 = np.logical_and(
        np.logical_or(
            ((Center[:,1]/Lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
            abs((Center[:,1]/Lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon),
        np.logical_or(
            (Center[:,0]/Lattice) % (2) < epsilon,
            abs((Center[:,0]/Lattice))% (2) < epsilon))

    Direction[JJ4]=-abs(Direction[JJ4])    

    JJ44 = np.logical_and( 
           np.logical_or(
          ((Center[:,1]/Lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
           abs((Center[:,1]/Lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon),
           np.logical_or(
          (Center[:,0]/Lattice +1) % (2) < epsilon,
           abs((Center[:,0]/Lattice +1)% (2) < epsilon)))

    Direction[JJ44]=abs(Direction[JJ44])

    return Direction    

def HoneycombSpinIceDirectionFerromagnetic(Center,Lattice,Direction):
    """In the ferromagnetic state all particles are oriented in one direction"""
    sqrt3 = np.sqrt(3)
    epsilon = 10**-6
        
    JJ1= np.logical_and( 
     np.logical_or(
        (Center[:,1]/Lattice) % (sqrt3) < epsilon,
         abs((Center[:,1]/Lattice)% (sqrt3) - sqrt3) < epsilon),
     np.logical_or(
        (Center[:,0]/Lattice -1/2) % (2) < epsilon,
         abs((Center[:,0]/Lattice -1/2))% (2) -(2) < epsilon))
                                            

    Direction[JJ1,1]=abs(Direction[JJ1,1])

    JJ11= np.logical_and( 
     np.logical_or(
         (Center[:,1]/Lattice) % (sqrt3) < epsilon,
          abs((Center[:,1]/Lattice)% (sqrt3) - sqrt3) < epsilon),
     np.logical_or(
         (Center[:,0]/Lattice +1/2) % 2 < epsilon,
          abs((Center[:,0]/Lattice +1/2)% 2 - 2) < epsilon))

    Direction[JJ11,1]=abs(Direction[JJ11,1])

     # JJ21 are the tilted spins below the even rows, on the left side of the cell. These should all point right.
    JJ21 =np.logical_and( 
      np.logical_or(
         ((Center[:,1]/Lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
          abs((Center[:,1]/Lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon),
      np.logical_or(
         (Center[:,0]/Lattice -1/4) % (2) < epsilon,
          abs((Center[:,0]/Lattice -1/4))% (2) < epsilon))

   
    # JJ5 are the spins in JJ21 that point left. These we flip.
    JJ5 = np.logical_and(JJ21, Direction[:,1]<0)
    Direction[JJ5] = -Direction[JJ5]

    # JJ22 are the tilted spins below the even rows, on the right side of the cell. These should all point left.
    JJ22 =np.logical_and( 
      np.logical_or(
         ((Center[:,1]/Lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
          abs((Center[:,1]/Lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon),
      np.logical_or(
         (Center[:,0]/Lattice -3/4) % (2) < epsilon,
          abs((Center[:,0]/Lattice -3/4))% (2) < epsilon))
    
    # JJ52 are the spins in JJ22 that point right. These we flip.
    JJ52 = np.logical_and(JJ22, Direction[:,1]<0)
    Direction[JJ52] = -Direction[JJ52]
    
    # JJ31 are the tilted spins above the even rows, on the left side of the cell. These should all point right.
    JJ31 =np.logical_and(
      np.logical_or(
        ((Center[:,1]/Lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon),
      np.logical_or(
         (Center[:,0]/Lattice -1/4) % (2) < epsilon,
          abs((Center[:,0]/Lattice -1/4))% (2) < epsilon))
    
    # JJ7 are the spins in JJ3 that point left. These we flip
    JJ7= np.logical_and(JJ31,Direction[:,1]<0)
    Direction[JJ7]=-Direction[JJ7]

   # JJ32 are the tilted spins above the even rows, on the left side of the cell. These should all point right.
    JJ32 =np.logical_and(
      np.logical_or(
        ((Center[:,1]/Lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((Center[:,1]/Lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon),
      np.logical_or(
         (Center[:,0]/Lattice -3/4) % (2) < epsilon,
          abs((Center[:,0]/Lattice -3/4))% (2) < epsilon))
    
    # JJ72 are the spins in JJ32 that point right. These we flip
    JJ72= np.logical_and(JJ31,Direction[:,1]<0)
    Direction[JJ72]=-Direction[JJ72]
    

   # JJ4 are vertical spins in odd rows. These should all point down.
    JJ4 = np.logical_and( 
        np.logical_or(
           ((Center[:,1]/Lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
           abs((Center[:,1]/Lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon),
        np.logical_or(
          (Center[:,0]/Lattice) % (2) < epsilon,
           abs((Center[:,0] / Lattice)% 2 - 2) < epsilon))

    Direction[JJ4,1]=abs(Direction[JJ4,1])    

    JJ44 = np.logical_and( 
           np.logical_or(
          ((Center[:,1]/Lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
           abs((Center[:,1]/Lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon),
           np.logical_or(
          (Center[:,0]/Lattice +1) % 2 < epsilon,
           abs((Center[:,0]/Lattice +1)% (2) - 2) < epsilon))

    Direction[JJ44,1]=abs(Direction[JJ44,1])

    return Direction