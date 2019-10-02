import numpy as np
import random
import scipy.spatial as spa
import copy as cp

def random_ordering(centers,directions,lattice):

    order_array = np.array([random.randrange(-1,2,2) for c in centers])
    
    return order_array<0

def square_ground_state(centers,directions,lattice):
    
    line = np.round((np.sum(centers,axis=1)/lattice-0.5)*1e6)*1e-6
    even = np.mod(line,2)<0.5
    odd = np.mod(line,2)>0.5

    dir_new = np.abs(directions)
    
    dir_new[even,1] = -1*dir_new[even,1]
    dir_new[odd,0] = -1*dir_new[odd,0]
    
    order_array = (np.round(np.sqrt(np.sum((dir_new-directions)**2,axis=1))*1e6)*1e-6).magnitude>0
    
    return order_array
    
def honeycomb_spin_solid(centers,directions,lattice):
    """ Honeycomb Spin Solid Phase """
    
    order_array = [True]*len(directions)
    
    directions = directions/lattice
    directions_0 = cp.deepcopy(directions)
    
    sqrt3 = np.sqrt(3)
    epsilon = 10**-6

    #To improve readibility, the quantities on this should be given names that are descriptive of their function. 
    
    # JJ1 are the vertical spins in even rows (starting from zero). One over 3 should all point down: those in 3*Lattice/2, 9*Lattice/2,  12*Lattice/2.
   
    JJ1= np.logical_or(
        (centers[:,1]/lattice) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice)% (sqrt3) - sqrt3) < epsilon)

    directions[JJ1]=abs(directions[JJ1])

    JJ12 = np.logical_and(JJ1,np.logical_or
                           ((centers[:,0]/lattice -3/2) % 3/2 < epsilon, 
                            abs((centers[:,0]/lattice -3/2) % 3/2 -3/2)<epsilon)) 

    directions[JJ12]=-directions[JJ12]

    # JJ2 are the tilted spins below the even rows. The 3 first (starting from 0) should point right, the 3 following should point left, ect..

    JJ2 = np.logical_or(
        ((centers[:,1]/lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
        abs((centers[:,1]/lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon)

    # JJ5 are the spins in JJ2 that should point right.

    JJ5=np.logical_and(JJ2,(((((4*centers[:,0])/lattice)-1 + epsilon)//2)//3 % 2) < epsilon)
        

    JJ7=np.logical_and(JJ5,directions[:,0]>0)

    directions[JJ7]=-directions[JJ7]

    # JJ9 are the spins in JJ2 that should point left.


    JJ9=np.logical_and(JJ2,np.logical_not((((((4*centers[:,0])/lattice)-1 + epsilon)//2)//3 % 2) < epsilon))

    JJ11=np.logical_and(JJ9,directions[:,0]<0)

    directions[JJ11]=-directions[JJ11]

    # JJ4 are vertical spins in odd rows. One over 3 should all point down: those in 0, 3*lattice,  6*lattice.

    JJ4 = np.logical_or(
        ((centers[:,1]/lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon)

    directions[JJ4]=abs(directions[JJ4])

    JJ42 = np.logical_and(JJ4, (centers[:,0]/lattice) % 3 < epsilon)
        
    directions[JJ42]=-directions[JJ42]

    # JJ3 are the tilted spins above the even rows. 
    JJ3 = np.logical_or(
        ((centers[:,1]/lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon)

    JJ6=np.logical_and(JJ3,(((((4*centers[:,0])/lattice)-1 + epsilon)//2)//3 % 2) < epsilon)

    JJ8=np.logical_and(JJ3,directions[:,0]<0)

    directions[JJ8]=-directions[JJ8]

    JJ10=np.logical_and(JJ3,np.logical_not(((((((4*centers[:,0])/lattice)-1 + epsilon)//2)//3 % 2) < epsilon)))
     
    JJ12=np.logical_and(JJ10,directions[:,0]>0)

    directions[JJ12]=-directions[JJ12]

    order_array = directions_0[:,0]!=directions[:,0]
    
    return order_array

def HoneycombSpinIcedirectionsBandAntiferromagnetic(directions,centers,lattice):
    """This state has lines in one direction in a ferromagnetic state with alternating orientation, and bands in the other directions in an antiferromagnetic state."""
    sqrt3 = np.sqrt(3)
    epsilon = 10**-6
    
    # JJ1 are the vertical spins in even rows (starting from zero). These should all point up. 
    JJ1= np.logical_or(
        (centers[:,1]/lattice) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice)% (sqrt3) - sqrt3) < epsilon)


    directions[JJ1,1]=abs(directions[JJ1,1])

   # JJ2 are the tilted spins below the even rows. These should all point right
    JJ2 = np.logical_or(
        ((centers[:,1]/lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
        abs((centers[:,1]/lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon)
    
    # JJ5 are the spins in JJ2 that point left. These we flip.
    JJ5 = np.logical_and(JJ2, directions[:,0]<0)
    directions[JJ5] = -directions[JJ5]
    
    # JJ3 are the tilted spins above the even rows. These should all point left
    JJ3 = np.logical_or(
        ((centers[:,1]/lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon)
    
    # JJ7 are the spins in JJ3 that point right. These we flip
    JJ7= np.logical_and(JJ3,directions[:,0]>0)
    directions[JJ7]=-directions[JJ7]

    # JJ4 are vertical spins in odd rows. These should all point down.
    JJ4 = np.logical_or(
        ((centers[:,1]/lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon)

    directions[JJ4]=-abs(directions[JJ4])    
    
    return directions

def HoneycombSpinIcedirectionsBandMixedAntiferromagnetic(centers,directions,lattice):
    """The state has the bands in one direction in a ferromagnetic state, and the bands in the other two directions in an antiferromagnetic state."""
    
    sqrt3 = np.sqrt(3)
    epsilon = 10**-6

    JJ1= np.logical_and( 
         np.logical_or(
        (centers[:,1]/lattice) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice)% (sqrt3) - sqrt3) < epsilon),
         np.logical_or(
        (centers[:,0]/lattice -1/2) % (2) < epsilon,
        abs((centers[:,0]/lattice -1/2))% (2) < epsilon))


    directions[JJ1,1]=abs(directions[JJ1,1])

    JJ11= np.logical_and( 
          np.logical_or(
         (centers[:,1]/lattice) % (sqrt3) < epsilon,
         abs((centers[:,1]/lattice)% (sqrt3) - sqrt3) < epsilon),
          np.logical_or(
         (centers[:,0]/lattice +1/2) % (2) < epsilon,
         abs((centers[:,0]/lattice +1/2))% (2) < epsilon))

    directions[JJ11,1]=-abs(directions[JJ11,1])

     # JJ2 are the tilted spins below the even rows. These should all point right
    JJ2 = np.logical_or(
        ((centers[:,1]/lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
        abs((centers[:,1]/lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon)
    
    # JJ5 are the spins in JJ2 that point left. These we flip.
    JJ5 = np.logical_and(JJ2, directions[:,0]<0)
    directions[JJ5] = -directions[JJ5]
    
    # JJ3 are the tilted spins above the even rows. These should all point left
    JJ3 = np.logical_or(
        ((centers[:,1]/lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon)
    
    # JJ7 are the spins in JJ3 that point right. These we flip
    JJ7= np.logical_and(JJ3,directions[:,0]>0)
    directions[JJ7]=-directions[JJ7]

   # JJ4 are vertical spins in odd rows. These should all point down.
    JJ4 = np.logical_and(
        np.logical_or(
            ((centers[:,1]/lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
            abs((centers[:,1]/lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon),
        np.logical_or(
            (centers[:,0]/lattice) % (2) < epsilon,
            abs((centers[:,0]/lattice))% (2) < epsilon))

    directions[JJ4]=-abs(directions[JJ4])    

    JJ44 = np.logical_and( 
           np.logical_or(
          ((centers[:,1]/lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
           abs((centers[:,1]/lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon),
           np.logical_or(
          (centers[:,0]/lattice +1) % (2) < epsilon,
           abs((centers[:,0]/lattice +1)% (2) < epsilon)))

    directions[JJ44]=abs(directions[JJ44])

    return directions    

def HoneycombSpinIcedirectionsFerromagnetic(centers,directions,lattice):
    """In the ferromagnetic state all particles are oriented in one direction"""
    sqrt3 = np.sqrt(3)
    epsilon = 10**-6
        
    JJ1= np.logical_and( 
     np.logical_or(
        (centers[:,1]/lattice) % (sqrt3) < epsilon,
         abs((centers[:,1]/lattice)% (sqrt3) - sqrt3) < epsilon),
     np.logical_or(
        (centers[:,0]/lattice -1/2) % (2) < epsilon,
         abs((centers[:,0]/lattice -1/2))% (2) -(2) < epsilon))
                                            

    directions[JJ1,1]=abs(directions[JJ1,1])

    JJ11= np.logical_and( 
     np.logical_or(
         (centers[:,1]/lattice) % (sqrt3) < epsilon,
          abs((centers[:,1]/lattice)% (sqrt3) - sqrt3) < epsilon),
     np.logical_or(
         (centers[:,0]/lattice +1/2) % 2 < epsilon,
          abs((centers[:,0]/lattice +1/2)% 2 - 2) < epsilon))

    directions[JJ11,1]=abs(directions[JJ11,1])

     # JJ21 are the tilted spins below the even rows, on the left side of the cell. These should all point right.
    JJ21 =np.logical_and( 
      np.logical_or(
         ((centers[:,1]/lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
          abs((centers[:,1]/lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon),
      np.logical_or(
         (centers[:,0]/lattice -1/4) % (2) < epsilon,
          abs((centers[:,0]/lattice -1/4))% (2) < epsilon))

   
    # JJ5 are the spins in JJ21 that point left. These we flip.
    JJ5 = np.logical_and(JJ21, directions[:,1]<0)
    directions[JJ5] = -directions[JJ5]

    # JJ22 are the tilted spins below the even rows, on the right side of the cell. These should all point left.
    JJ22 =np.logical_and( 
      np.logical_or(
         ((centers[:,1]/lattice)+ (sqrt3/4)) % (sqrt3) < epsilon, 
          abs((centers[:,1]/lattice +sqrt3/4) % sqrt3 - sqrt3) < epsilon),
      np.logical_or(
         (centers[:,0]/lattice -3/4) % (2) < epsilon,
          abs((centers[:,0]/lattice -3/4))% (2) < epsilon))
    
    # JJ52 are the spins in JJ22 that point right. These we flip.
    JJ52 = np.logical_and(JJ22, directions[:,1]<0)
    directions[JJ52] = -directions[JJ52]
    
    # JJ31 are the tilted spins above the even rows, on the left side of the cell. These should all point right.
    JJ31 =np.logical_and(
      np.logical_or(
        ((centers[:,1]/lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon),
      np.logical_or(
         (centers[:,0]/lattice -1/4) % (2) < epsilon,
          abs((centers[:,0]/lattice -1/4))% (2) < epsilon))
    
    # JJ7 are the spins in JJ3 that point left. These we flip
    JJ7= np.logical_and(JJ31,directions[:,1]<0)
    directions[JJ7]=-directions[JJ7]

   # JJ32 are the tilted spins above the even rows, on the left side of the cell. These should all point right.
    JJ32 =np.logical_and(
      np.logical_or(
        ((centers[:,1]/lattice) - (sqrt3/4)) % (sqrt3) < epsilon,
        abs((centers[:,1]/lattice - sqrt3/4) % sqrt3 - sqrt3) < epsilon),
      np.logical_or(
         (centers[:,0]/lattice -3/4) % (2) < epsilon,
          abs((centers[:,0]/lattice -3/4))% (2) < epsilon))
    
    # JJ72 are the spins in JJ32 that point right. These we flip
    JJ72= np.logical_and(JJ31,directions[:,1]<0)
    directions[JJ72]=-directions[JJ72]
    

   # JJ4 are vertical spins in odd rows. These should all point down.
    JJ4 = np.logical_and( 
        np.logical_or(
           ((centers[:,1]/lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
           abs((centers[:,1]/lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon),
        np.logical_or(
          (centers[:,0]/lattice) % (2) < epsilon,
           abs((centers[:,0] / lattice)% 2 - 2) < epsilon))

    directions[JJ4,1]=abs(directions[JJ4,1])    

    JJ44 = np.logical_and( 
           np.logical_or(
          ((centers[:,1]/lattice)+ (sqrt3/2)) % (sqrt3) < epsilon,
           abs((centers[:,1]/lattice + sqrt3/2) % sqrt3 - sqrt3) < epsilon),
           np.logical_or(
          (centers[:,0]/lattice +1) % 2 < epsilon,
           abs((centers[:,0]/lattice +1)% (2) - 2) < epsilon))

    directions[JJ44,1]=abs(directions[JJ44,1])

    return directions