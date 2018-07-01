from icenumerics.spins import *
from icenumerics.colloidalice import colloidal_ice
import subprocess # Subprocess is a default library which allows us to call a command line program from within a python script
import shutil # shutil allows us to move files around. This is usefull to organize the resulting input and output files. 
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial as sptl
import pandas as pd

def spin_crossing_point(S1,S2):
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

def unique_points(Points,Tol = 0.1):
    """Returns only the distinct points (with a tolerance)."""
    Distance = sptl.distance.squareform(sptl.distance.pdist(Points))
    IsLast = []
    for i,p in enumerate(Points):
        IsLast = IsLast + [not np.any(Distance[(i+1):,i]<=Tol)]
        
    return Points[np.array(IsLast),:]
    
def colloidal_ice_vector(C):
    """Extracts an array of centers and directions from a Colloidal Ice System"""
    Vectors = np.array(np.zeros(len(C)),dtype=[('Center',np.float,(2,)),('Direction',np.float,(2,))])
    i=0
    for c in C:
        Vectors[i] = (c.center[0:2],c.direction[0:2])
        i=i+1
    return Vectors
    
def spin_ice_vector(S):
    """Extracts an array of centers and directions from a Spin Ice System"""
    Vectors = np.array(np.zeros(len(S)),dtype=[('Center',np.float,(2,)),('Direction',np.float,(2,))])
    i=0
    for s in S:
        Vectors[i] = (s.center[0:2],s.direction[0:2])
        i=i+1
    return Vectors
        
def calculate_neighbor_pairs(Centers):
    """This function makes a list of all the Pairs of Delaunay Neighbors from an array of points"""
    
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

def from_neighbors_get_nearest_neighbors(NeighborPairs):
    # This function takes a list of Delaunay Neighbor Pairs and returns only those which are close to the minimum distance.
    NeighborPairs['Distance']=np.around(NeighborPairs['Distance'],decimals=4)
    NeighborPairs = NeighborPairs[NeighborPairs['Distance']<=np.min(NeighborPairs['Distance'])*1.1]
    
    return NeighborPairs

def get_vertices_positions(NeighborPairs,spins):
    # From a list of Spins, get neighboring spins, and get the crossing point of each, which defines a vertex.  
    for i,n in enumerate(NeighborPairs):
        NeighborPairs[i]['Vertex'] = spin_crossing_point(spins[n['Pair'][0]],spins[n['Pair'][1]])[0:2]
    
    return NeighborPairs

class vertices():
    def __init__(self):
        # This function initializes the Vertices Array
        self.array=np.array([],
                      dtype=[
                          ('Location', float,(2,)),
                          ('id',int),
                          ('Coordination',int),
                          ('Charge',int),
                          ('Dipole',int,(2,))])
        
    def colloids_to_vertices(self,C):

        self.spins = colloidal_ice_vector(C)
        
        NeighborPairs = calculate_neighbor_pairs(self.spins['Center'])

        NeighborPairs = from_neighbors_get_nearest_neighbors(NeighborPairs)

        NeighborPairs = get_vertices_positions(NeighborPairs,self.spins)
        
        v = unique_points(NeighborPairs['Vertex'])
        
        ## Make Vertex array
        self.array=np.array(np.empty(np.shape(v)[0]),
                            dtype=[
                                ('Location', float,(2,)),
                                ('id',int),
                                ('Coordination',int),
                                ('Charge',int),
                                ('Dipole',int,(2,))])
        
        self.array['Location'] = v
        
        ## Make Neighbors directory
        self.neighbors = {}

        for i,v in enumerate(self.array):
            v['id']=i
            self.neighbors[i] = []
            for n in NeighborPairs:
                if sptl.distance.euclidean(n['Vertex'],v['Location'])<np.mean(n['Vertex']*1e-6):
                    self.neighbors[i]=self.neighbors[i]+list(n['Pair'])
            self.neighbors[i] = set(self.neighbors[i])
            
            ## Calculate Coordination
            v['Coordination'] = len(self.neighbors[i])
        
            ## Calculate Charge and Dipole
            v['Charge'] = 0
            v['Dipole'] = [0,0]
               
            for n in self.neighbors[v['id']]:
                v['Charge']=v['Charge'] + np.sign(np.sum((v['Location']-self.spins[n]['Center'])*self.spins[n]['Direction']))

                v['Dipole']=v['Dipole'] + self.spins[n]['Direction']
         
        return self
        
    def spins_to_vertices(self,s):

        self.spins = spin_ice_vector(C)
        
        NeighborPairs = calculate_neighbor_pairs(self.spins['Center'])

        NeighborPairs = from_neighbors_get_nearest_neighbors(NeighborPairs)

        NeighborPairs = get_vertices_positions(NeighborPairs,self.spins)
        
        v = unique_points(NeighborPairs['Vertex'])
        
        ## Make Vertex array
        self.array=np.array(np.empty(np.shape(v)[0]),
                            dtype=[
                                ('Location', float,(2,)),
                                ('id',int),
                                ('Coordination',int),
                                ('Charge',int),
                                ('Dipole',int,(2,))])
        
        self.array['Location'] = v
        
        ## Make Neighbors directory
        self.neighbors = {}

        for i,v in enumerate(self.array):
            v['id']=i
            self.neighbors[i] = []
            for n in NeighborPairs:
                if sptl.distance.euclidean(n['Vertex'],v['Location'])<np.mean(n['Vertex']*1e-6):
                    self.neighbors[i]=self.neighbors[i]+list(n['Pair'])
            self.neighbors[i] = set(self.neighbors[i])
            
            ## Calculate Coordination
            v['Coordination'] = len(self.neighbors[i])
        
            ## Calculate Charge and Dipole
            v['Charge'] = 0
            v['Dipole'] = [0,0]
               
            for n in self.neighbors[v['id']]:
                v['Charge']=v['Charge'] + np.sign(np.sum((v['Location']-self.spins[n]['Center'])*self.spins[n]['Direction']))

                v['Dipole']=v['Dipole'] + self.spins[n]['Direction']
         
        return self
        
    def display(self,ax = False,DspCoord = False):
                
        if not ax:
            fig1, ax = plt.subplots(1,1)  

        if not DspCoord:
            for v in self.array:
                if v['Charge']>0:
                    c = 'r'
                else:
                    c = 'b'
                ax.add_patch(patches.Circle(
                    (v['Location'][0],v['Location'][1]),radius = abs(v['Charge'])*2,
                    ec='none', fc=c))
                X = v['Location'][0]
                Y = v['Location'][1]
                if v['Charge']==0:
                    DX = v['Dipole'][0]
                    DY = v['Dipole'][1]
                    ax.add_patch(patches.Arrow(X-DX,Y-DY,2*DX,2*DY,width=5,fc='k'))
                
        if DspCoord: 
            for v in self.array:
                if v['Charge']>0:
                    c = 'r'
                else:
                    c = 'b'
                    
                ax.add_patch(patches.Circle(
                    (v['Location'][0],v['Location'][1]),radius = abs(v['Coordination'])*2,
                    ec='none', fc=c))
                X = v['Location'][0]
                Y = v['Location'][1]
        
        #ax.set_aspect("equal")    
        #plt.axis("equal")

