import numpy as np
import warnings
from IceNumerics.Vector import Vector

class TrapGeometry():
    # The trap geometry object contains:
    # TrapSep: Distance between stable points of the traps [in nm]
    # Height: Hill height of the trap [in nm]
    # Stiffness: Stiffness of the trap [in pNnm].
    
    # Trap Separation could be determined also by the spin dipole.
    # I need to think how to put them together. 

    def __init__(self,**kargs):
        self.trap_sep_ratio = float(1); # The trap sep ratio is the ratio between the spin dipole and the final trap separation.
        self.height = 200;
        self.stiffness = 1.2e-4;
        self.stiffness_spread = 0;
        if 'TrapSepRatio' in kargs: self.trap_sep_ratio = float(kargs['TrapSepRatio'])
        if 'Height' in kargs: self.height = float(kargs['Height'])
        if 'Stiffness' in kargs: self.stiffness = float(kargs['Stiffness'])
        if 'Stiffness_Spread' in kargs:
            self.stiffness_spread = float(kargs['Stiffness_Spread'])

class ColloidParameters():
    # Colloid Parameters are:
    # Susceptibility [adimensional]
    # Mass [g]
    # Diffusion [nm^2/s]
    def __init__(self,**kargs):
        self.susceptibility = 0.0576
        self.diffusion = 0.125e6 # in nm^2/s
        self.diameter = 10.3e3 #nm
        self.rel_density = 1.9 #nm
        self.volume = 4/3*np.pi*(self.diameter/2)**3

        if 'Susceptibility' in kargs: self.susceptibility = kargs['Susceptibility']
        if 'Diffusion' in kargs: self.diffusion = kargs['Diffusion']
        if 'Diameter' in kargs:
            self.diameter = kargs['Diameter']
            self.volume = 4/3*np.pi*(self.diameter/2)**3
            if 'Volume' in kargs or 'Radius' in kargs:
                warn('You have too many particle size specifications')
        elif 'Radius' in kargs:
            self.diameter = 2*kargs['Diameter']
            if 'Volume' in kargs:
                warn('You have too many particle size specifications')
        elif 'Volume' in kargs:
            self.volume = kargs['Volume']
            self.diameter = np.power((3*self.volume/4/np.pi),1/3)
        if 'RelativeDensity' in kargs:
            self.rel_density = kargs['RelativeDensity']

        damp = 0.03e-6 #s
        temperature = float(300) #K 
        kb = float(4/300) #pN nm

        if 'Damp' in kargs: damp = kargs['Damp']
        if 'Temperature' in kargs: temperature = kargs['Temperature']
        if 'Kb' in kargs: kb = kargs['Kb']
            
        self.mass = kb*temperature*damp/self.diffusion
        
class WorldParameters():
    # World Parameters contains:
    # Region [in nm]: Default is calculated by set_region(ColloidalIce)
    # Periodic 
    # Temperature [K]
    # Kb [pN nm / K]
    # FieldZ
    # Bias
    
    def __init__(self,**kargs):

        self.region = np.array([0,0,0,0,0,0]);
        self.periodic = False
        self.temperature = float(300)
        self.kb = float(4/300) 
        self.fieldz = [30,0] #mT, s
        self.bias = Vector([0,0,0]) #pN
        self.gravity = 9.8e9 #pN/g = nm/s^2
        self.medium_density = 1e-21 #g/nm^3
        self.permeability = 4e5*np.pi #pN/A^2
        self.damp = 0.03e-6 #s
                
        if 'Region' in kargs: self.region = np.array(kargs['Region'])
        if 'Periodic' in kargs: self.periodic = kargs['Periodic']
        if 'Temperature' in kargs: self.temperature = kargs['Temperature']
        if 'Kb' in kargs: self.kb = kargs['Kb']
        if 'FieldZ' in kargs: self.fieldz = kargs['FieldZ']
        if 'Bias' in kargs: self.bias = kargs['Bias']
        if 'Gravity' in kargs: self.gravity = kargs['Gravity']
        if 'Medium_Density' in kargs: self.medium_density = kargs['Medium_Density']
        if 'Permeability' in kargs: self.permeability = kargs['Permeability']
        if 'Damp' in kargs: self.damp = kargs['Damp']

    def set_region(self,ColloidalIce,**kargs):
        if 'Periodic' in kargs: self.periodic = kargs['Periodic']
        if not self.periodic:
            self.region = np.array([0,0,0,0,0,0],dtype=np.float64)
            for c in ColloidalIce:
                
                if ColloidalIce[c].center[0]<self.region[0]:
                    self.region[0]=ColloidalIce[c].center[0]
                elif ColloidalIce[c].center[0]>self.region[1]:
                    self.region[1]=ColloidalIce[c].center[0]
                    
                if ColloidalIce[c].center[1]<self.region[2]:
                    self.region[2]=ColloidalIce[c].center[1]
                elif ColloidalIce[c].center[1]>self.region[3]:
                    self.region[3]=ColloidalIce[c].center[1]
                    
                if ColloidalIce[c].center[2]<self.region[4]:
                    self.region[4]=ColloidalIce[c].center[2]
                elif ColloidalIce[c].center[2]>self.region[5]:
                    self.region[5]=ColloidalIce[c].center[2]

        else:
            error('I still don\'t know how to calculate periodic region')
        self.region += np.array([-1,1,-1,1,-0.05,0.05])*ColloidalIce.lattice


class SimulationParameters():
    def __init__(self,**kargs):

        self.seed = 1
        self.timestep = 10e-6 #seconds/step
        self.runs = 1
        self.run_time = 30
        self.thermo = 1e5
        self.framerate = 15 #frames/second
        self.filename = "LAMMPSTest"
        self.timestamp = True
        self.targetdir = ''
        
        if 'Seed' in kargs: self.seed = np.array(kargs['Seed'])
        if 'Timestep' in kargs: self.timestep = kargs['Timestep']
        if 'Runs' in kargs: self.runs = kargs['Runs']
        if 'Time' in kargs: self.run_time = kargs['Time']
        if 'Thermo' in kargs: self.thermo = kargs['Thermo']
        if 'Framerate' in kargs: self.framerate = kargs['Framerate']
        if 'Filename' in kargs: self.filename = kargs['Filename']
        if 'Timestamp' in kargs: self.timestamp = kargs['Timestamp']
        if 'TargetDir' in kargs: self.targetdir = kargs['TargetDir']
