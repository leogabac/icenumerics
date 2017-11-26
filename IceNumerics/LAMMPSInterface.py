import random
from IceNumerics.Spins import Vector
import numpy as np
import time
import os
import sys
import shutil # shutil allows us to move files around. This is usefull to organize the resulting input and output files. 

class LAMMPSScript():
    def __init__(self,colloidalice,simparameters,test=False):
        ## I need as many atoms as simulation runs. The atoms corresponding to
        ## traps can be grouped by region
        self.filename = simparameters.filename
        if test:
            self.filename = simparameters.filename
        else:
            if simparameters.timestamp:
                self.filename = simparameters.filename + \
                                         time.strftime('_%Y_%m_%d_%H_%M_%S')
                    
        self.simparameters = simparameters
        
        ArbitColloid = next(iter(colloidalice.values()))

        self.write_input_file(colloidalice.worldparams,simparameters,ArbitColloid)
        self.write_data_file(colloidalice,simparameters)
        
    def write_data_file(self,ColloidalIce,SimParameters):
        
        filename = self.filename
        f = open(filename+'.data','w')
        
        runs = SimParameters.runs

        Cutoff = 2e-3*ColloidalIce.lattice
 
        f.write("This is the initial atom setup of %s \n"%filename)
        f.write("%d atoms\n"%(len(ColloidalIce)*(SimParameters.runs+1)))
        f.write("%d atom types\n"%(SimParameters.runs+1))
        f.write("%d bonds\n"%(len(ColloidalIce)*(SimParameters.runs)))
        f.write("%d bond types\n"%(len(ColloidalIce)*(SimParameters.runs)))
        f.write(("%2.2f %2.2f xlo xhi \n"+
                "%2.2f %2.2f ylo yhi \n"+
                "%2.2f %2.2f zlo zhi \n") %
                tuple(a*1e-3 for a in ColloidalIce.worldparams.region))

        ## TRAPS DEFINITIONS
        f.write("\nAtoms\n\n")
        trapID = 1
        TrapsDirectory = {}
        for c in ColloidalIce:
            write_trap(f,ColloidalIce[c],trapID,runs+1)
            TrapsDirectory[trapID] = c
            trapID+=1

        atomID = trapID
        AtomID2TrapDirectory = {}
        
        for run in range(0,runs):
            for t in TrapsDirectory:
                c = TrapsDirectory[t]
                write_colloid(f,ColloidalIce[c],
                              atomID,run+1,ColloidalIce.worldparams,t)
                AtomID2TrapDirectory[atomID] = t;
                atomID+=1

        ## BOND DEFINITIONS

        f.write("\nBonds\n\n")

        BondID = 1
        BondDirectory={}
        for atom in AtomID2TrapDirectory:
            trap = AtomID2TrapDirectory[atom]
            bond_type = BondID
            f.write("%d %d %d %d\n"%(BondID,bond_type,trap,atom))
            BondDirectory[BondID] = atom
            BondID +=1

        f.write("\nBond Coeffs\n\n")

        for bond in BondDirectory:
            bond_type = bond
            atom = BondDirectory[bond]
            c = TrapsDirectory[AtomID2TrapDirectory[atom]]
            
            weight = \
                (ColloidalIce[c].colloidparams.volume * \
                (ColloidalIce[c].colloidparams.rel_density-1) * \
                ColloidalIce.worldparams.medium_density * \
                ColloidalIce.worldparams.gravity)*1e-3
            trap_sep = ColloidalIce[c].direction.magnitude()*1e-3
            stiffness = ColloidalIce[c].geometry.stiffness*1000 * weight
            stiffnesshill = \
                (1+random.gauss(0,1)*ColloidalIce[c].geometry.stiffness_spread)* \
                8/np.power(trap_sep,2)*weight*ColloidalIce[c].geometry.height/1000

            f.write("%d %f %f\n"%(bond_type,stiffness,stiffnesshill))

                
        ## PAIR COEFFICIENT DEFINITIONS

        f.write("\nPairIJ Coeffs\n\n")
        Cutoff = 2e-3*ColloidalIce.lattice

        for I in range(0,runs+1):
            for J in range(I,runs+1):
                f.write("%d %d "%(I+1,J+1))
                if I==J and not J==(runs): f.write("0.0 0.0 0.0 %f\n"%(Cutoff))
                else: f.write("0.0 0.0 0.0 0.0 \n")
            
    def write_input_file(self,worldparams,simparams,ArbitColloid):
        filename = self.filename

        # This specifies the parameters of the experiment
        # damp is converted from seconds to microseconds
        damp = worldparams.damp * 1e6
        temp = worldparams.temperature
        kb = worldparams.kb
        seed = simparams.seed
        #Sampling [steps/frame] = 1/((frames/second)*(seconds/step))
        sampling = round(1/simparams.timestep/simparams.framerate)
        run_steps = round(simparams.run_time/simparams.timestep)
        
        moment = (ArbitColloid.colloidparams.susceptibility *
                  worldparams.fieldz[0]*ArbitColloid.colloidparams.volume /
                  worldparams.permeability)/2.99e8

        fieldh = (worldparams.fieldz[0]/worldparams.permeability)/      2.99e8*1e9
                  
        f = open(filename+'.CI','w')
        f.write("units micro\n")
        f.write("atom_style hybrid sphere paramagnet bond\n")

        ## This is an option that should be specified. The default is open
        if not worldparams.periodic:
            f.write("boundary s s p\n")
        else:
            f.write("boundary p p p\n")

        f.write("dimension 2\n")
        f.write("neighbor 4.0 nsq\n")
        f.write("pair_style lj/cut/dipole/cut 20\n")
        f.write("bond_style biharmonic \n")

        f.write("\n#----Read Data---#\n\n")
        f.write("read_data %s\n"%(filename+'.data'))
        f.write("\n#----End Read Data---#\n\n")

        f.write("group Atoms type %d:%d\n"%(1,simparams.runs))
        f.write("mass * 1\n")

        f.write("\n#----Fix Definitions---#\n\n")
        f.write("variable Bmax atom %f\n" % fieldh)
        f.write("variable Tmax atom %e\n" % worldparams.fieldz[1])
        f.write("variable field atom ")
        if worldparams.fieldz[1]!=0:
            f.write(    "v_Bmax-(abs((v_Bmax/v_Tmax*(time-v_Tmax)))-(v_Bmax/v_Tmax*(time-v_Tmax)))/2\n\n")
        else:
            f.write("v_Bmax")
            
        f.write("fix \t1 Atoms bd %f %f %d\n"%(temp,damp,seed))
        f.write("fix \t2 all enforce2d\n")
        f.write("fix \t3 Atoms addforce %f %f %f\n"%tuple(worldparams.bias*1e-3))
        f.write("fix \t4 Atoms setdipole 0 0 v_field\n")
        f.write("\n#----End of Fix Definitions---#\n")

        f.write("#----Run Definitions---#\n\n")
        f.write("timestep \t%d\n"%(simparams.timestep*1e6))
        f.write("dump \t3 all custom %d %s.lammpstrj id type x y z mu\n" %
                (sampling,filename))
        f.write("thermo_style \tcustom step atoms\n")
        f.write("thermo \t%d\n"%simparams.thermo)
        f.write("run \t%d\n"%run_steps)
        f.close()
        
    def LAMMPSRun(self):
        TargetDir = self.simparameters.targetdir
        if not os.path.exists(TargetDir):
            os.makedirs(TargetDir)

        if not os.path.exists(os.path.join(TargetDir,"Data and Run Files")):
            os.makedirs(os.path.join(TargetDir,"Data and Run Files"))

        if sys.platform=='darwin':
            LAMMPSExec = "./IceNumerics/lmp_mac"
        else:
            LAMMPSExec = "IceNumerics\lmp_mingw64.exe"
            
        os.system(LAMMPSExec + " -in "+self.filename+".CI")
        
        self.lammpstrj = os.path.join(TargetDir,self.filename+".lammpstrj")
        self.lammpsinput = os.path.join(TargetDir,"Data and Run Files",self.filename+".CI")
        self.lammpsdat = os.path.join(TargetDir,"Data and Run Files",self.filename+".data")
        
        shutil.move(self.filename+".CI",self.lammpsinput)
        shutil.move(self.filename+".data",self.lammpsdat)
        shutil.move(self.filename+".lammpstrj",self.lammpstrj)
        
        return self


def write_trap(File,Colloid,trapID,traps_type):
    
        trap_sep = Colloid.direction.magnitude()*1e-3
        diameter = Colloid.colloidparams.diameter * 1e-3
        density = Colloid.colloidparams.mass / \
                  Colloid.colloidparams.volume * 1e21
        File.write("%6.0d\t%d\t"%(trapID,traps_type))
        File.write("%5.2f\t%5.2f\t%5.6f\t" %
                 tuple(Colloid.center * 1e-3+
                      Vector((0,0,0))))
        File.write("%5.2f\t%5.2g\t"%(diameter,density))
        File.write("0.0\t%5.2g\t%5.2g\t%5.2g\t0.0\t"%
                tuple(Colloid.direction.unit()*
                       (trap_sep)))
        File.write("%d\n"%trapID)

def write_colloid(File,Colloid,atomID,atom_type,world,trap):
    
        trap_sep = Colloid.direction.magnitude()*1e-3
        diameter = Colloid.colloidparams.diameter * 1e-3
        density = Colloid.colloidparams.mass / \
                  Colloid.colloidparams.volume * 1e21
        # moment = (Colloid.colloidparams.susceptibility *
                  # world.fieldz*Colloid.colloidparams.volume /
                  # world.permeability)/2.99e8
        moment = 0
        susceptibility = Colloid.colloidparams.susceptibility
        if Colloid.ordering=="Random": order_flip = random.randrange(-1,2,2)
        else: order_flip = 1;
        
        File.write("%6.0d\t%d\t"%(atomID,atom_type))
        File.write("%5.2f\t%5.2f\t%5.6f\t" %
                tuple((Colloid.center +
                      Colloid.colloid*order_flip)*1e-3 +
                      Vector((
                          0,
                          0,
                          random.gauss(0,1)))*0.01
                      ))
        File.write("%f\t%f\t"%(diameter,density))
        File.write("0.0\t0.0\t0.0\t0.0\t%5.5g\t"%susceptibility)
        File.write("%d\n"%trap)                

class LazyOpenLAMMPSTrj():
    def __init__(self,Filename):
        self.T = dict([])
        self.Name = Filename
        item = dict([])
        with open(Filename) as d:
            line = "d"
            while line:
                line = d.readline()
                
                if 'ITEM: TIMESTEP' in line:
                    line = d.readline()
                    t = float(line)
                    
                if 'ITEM: NUMBER OF ATOMS' in line:
                    line = d.readline()
                    item["atoms"] = float(line)
                    
                if 'ITEM: ATOMS' in line:
                    item["location"] = d.tell()
                    self.T[t] = item
                
    def readframe(self,time):
        Atoms = np.empty([6,int(self.T[time]["atoms"])])
        j=0
        with open(self.Name) as d:
            d.seek(self.T[time]["location"])
            for i in range(0,int(self.T[time]["atoms"])):
                line = d.readline()
                Atoms[:,j]=np.array([float(i) for i in line.split(' ') if i!='\n'])
                j=j+1;
        return Atoms