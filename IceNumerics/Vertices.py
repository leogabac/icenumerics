
# coding: utf-8

# In[1]:


from IceNumerics.Spins import *
from IceNumerics.ColloidalIce import ColloidalIce
from IceNumerics.LAMMPSInterface import *
import subprocess # Subprocess is a default library which allows us to call a command line program from within a python script
import shutil # shutil allows us to move files around. This is usefull to organize the resulting input and output files. 
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial as sptl
import pandas as pd

def FindCrossingOfSpinVectors(S1,S2):
    # This works well in 2d. In 3d it's triciker
    if not (S1['Direction']==S2['Direction']).all():
        A = np.ones([2,2])
        A[:,0] = S1['Direction']
        A[:,1] = -S2['Direction']

        b = np.array([
            S2['Center'][0]-S1['Center'][0],
            S2['Center'][1]-S1['Center'][1]])

        lam = np.linalg.solve(A,b)

        return S1['Center']+lam[0]*S1['Direction']
    else:
        return np.Inf+np.zeros(np.shape(S1['Center']))

def UniquePoints(Points,Tol = 0.1):
    # This function returns only the distinct points (with a tolerance). 
    Distance = sptl.distance.squareform(sptl.distance.pdist(Points))
    IsLast = []
    for i,p in enumerate(Points):
        IsLast = IsLast + [not np.any(Distance[(i+1):,i]<=Tol)]
        
    return Points[np.array(IsLast),:]
    
def ColloidsToVector(C):
    # Extracts an array of centers and directions from a Colloidal Ice System
    Vectors = np.array(np.zeros(len(C)),dtype=[('Center',np.float,(2,)),('Direction',np.float,(2,))])
    i=0
    for c in C:
        Vectors[i] = (C[c].center[0:2],C[c].direction[0:2])
        i=i+1
    return Vectors
        
def CalculateNeighborPairs(Centers):
    # This function makes a list of all the Pairs of Delaunay Neighbors from an array of points
    
    tri = sptl.Delaunay(Centers)

    # List all Delaunay neighbors in the system
    NeighborPairs = np.array(np.zeros(2*np.shape(tri.simplices)[0]),
                             dtype=[('Pair',np.int,(2,)),('Distance',np.float),('Vertex',np.float,(2,))])

    i = 0
    for t in tri.simplices:
        NeighborPairs[i]['Pair'] = np.sort(t[0:2])
        NeighborPairs[i]['Distance'] = sptl.distance.euclidean(Centers[t[0]],Centers[t[1]])
        NeighborPairs[i+1]['Pair'] = np.sort(t[1:3])
        NeighborPairs[i+1]['Distance'] = sptl.distance.euclidean(Centers[t[1]],Centers[t[2]])
        i = i+2

    return NeighborPairs

def FromNeigbhorsGetNearestNeighbors(NeighborPairs):
    # This function takes a list of Delaunay Neighbor Pairs and returns only those which are close to the minimum distance.
    NeighborPairs['Distance']=np.around(NeighborPairs['Distance'],decimals=4)
    NeighborPairs = NeighborPairs[NeighborPairs['Distance']<=np.min(NeighborPairs['Distance'])*1.1]
    
    return NeighborPairs

def GetVerticesPositions(NeighborPairs,Spins):
    # From a list of Spins, get neighboring spins, and get the crossing point of each, which defines a vertex.  
    for i,n in enumerate(NeighborPairs):
        NeighborPairs[i]['Vertex'] = FindCrossingOfSpinVectors(Spins[n['Pair'][0]],Spins[n['Pair'][1]])[0:2]
    
    return NeighborPairs

class Vertices():
    def __init__(self):
        # This function initializes the Vertices Array
        self.array=np.array([],
                      dtype=[
                          ('Location', float,(2,)),
                          ('id',int),
                          ('Coordination',int),
                          ('Charge',int),
                          ('Dipole',int,(2,))])
        
    def ColloidsToVertices(self,C):

        self.Spins = ColloidsToVector(C)
        
        NeighborPairs = CalculateNeighborPairs(self.Spins['Center'])

        NeighborPairs = FromNeigbhorsGetNearestNeighbors(NeighborPairs)

        NeighborPairs = GetVerticesPositions(NeighborPairs,self.Spins)
        
        V = UniquePoints(NeighborPairs['Vertex'])
        
        ## Make Vertex array
        self.array=np.array(np.empty(np.shape(V)[0]),
                            dtype=[
                                ('Location', float,(2,)),
                                ('id',int),
                                ('Coordination',int),
                                ('Charge',int),
                                ('Dipole',int,(2,))])
        
        self.array['Location'] = V
        
        ## Make Neighbors directory
        self.Neighbors = {}

        for i,V in enumerate(self.array):
            V['id']=i
            self.Neighbors[i] = []
            for N in NeighborPairs:
                if sptl.distance.euclidean(N['Vertex'],V['Location'])<np.mean(N['Vertex']*1e-6):
                    self.Neighbors[i]=self.Neighbors[i]+list(N['Pair'])
            self.Neighbors[i] = set(self.Neighbors[i])
            
            ## Calculate Coordination
            V['Coordination'] = len(self.Neighbors[i])
        
            ## Calculate Charge and Dipole
            V['Charge'] = 0
            V['Dipole'] = [0,0]
               
            for n in self.Neighbors[V['id']]:
                V['Charge']=V['Charge'] + np.sign(np.sum((V['Location']-self.Spins[n]['Center'])*self.Spins[n]['Direction']))

                V['Dipole']=V['Dipole'] + self.Spins[n]['Direction']
                                    
        return self
        
    def display(self,DspObj = False,DspCoord = False):

        d=0.1
        AxesLocation = [d,d,1-2*d,1-2*d]
                
        if not DspObj:
            fig1 = plt.figure(1)
            ax1 = plt.axes(AxesLocation)
        elif DspObj.__class__.__name__ == "Figure":
            fig1 = DspObj
            ax1 = plt.axes(AxesLocation, frameon=0)
            fig1.patch.set_visible(False)
        elif DspObj.__class__.__name__ == "Axes":
            ax1 = DspObj
            fig1.patch.set_visible(False)
        elif DspObj.__class__.__name__ == "AxesSubplot":
            ax1 = DspObj

        if not DspCoord:
            for v in self.array:
                if v['Charge']>0:
                    c = 'r'
                else:
                    c = 'b'
                ax1.add_patch(patches.Circle(
                    (v['Location'][0],v['Location'][1]),radius = abs(v['Charge'])*2000,
                    ec='none', fc=c))
                X = v['Location'][0]
                Y = v['Location'][1]
                if v['Charge']==0:
                    DX = v['Dipole'][0]*2e-1
                    DY = v['Dipole'][1]*2e-1
                    ax1.add_patch(patches.Arrow(X-DX,Y-DY,2*DX,2*DY,width=7e3,fc='k'))
                
        if DspCoord: 
            for v in self.array:
                ax1.add_patch(patches.Circle(
                    (v['Location'][0],v['Location'][1]),radius = abs(v['Coordination'])*2000,
                    ec='none', fc=c))
                X = v['Location'][0]
                Y = v['Location'][1]
            
        plt.axis("equal")

        if DspObj.__class__.__name__ == "Figure":
            ax1 = DspObj
            fig1.patch.set_visible(False) 
            plt.show(block = True)    
                

