from icenumerics.spins import *
from icenumerics.colloidalice import colloidal_ice
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial as sptl
import pandas as pd

def unwrap_trj(trj,bounds):
    """ Unwraps trj around periodic boundaries"""
    trj2 = trj.copy(deep=True)

    def unwrap(p):
        p.iloc[:] = np.unwrap(p,axis=0)
        return p

    for c in trj.columns:
        trj2[c] = (trj2[c] - bounds[c+"_min"].values)/(bounds[c+"_max"].values - bounds[c+"_min"].values)

    trj2 = (trj2*2*np.pi).groupby("id").apply(unwrap)/(2*np.pi)

    for c in trj.columns:
        trj2[c] = trj2[c]*(bounds[c+"_max"].values - bounds[c+"_min"].values) + bounds[c+"_min"].values

    return trj2

def unwrap_frame(col_trj,bnd,axis):
    box_size = (bnd[axis+"_max"]-bnd[axis+"_min"]).values[0]
    mask = col_trj.loc[:,"c"+axis]>box_size/2
    col_trj.loc[mask,"c"+axis] = col_trj.loc[mask,"c"+axis]-box_size

    mask = col_trj.loc[:,"c"+axis]<-box_size/2
    col_trj.loc[mask,"c"+axis]= col_trj.loc[mask,"c"+axis]+box_size

def get_ice_trj(trj,bounds):
    """ Converts lammps trj to ice trj"""
    # in the trj dataframe, traps and atoms are labeled by different types
    traps = trj[trj.type==2].copy(deep=True)
    traps = traps.rename(columns = {"mux":"dx","muy":"dy","muz":"dz"})
    atoms = trj[trj.type==1].copy(deep=True)
    atoms = unwrap_trj(atoms.filter(["x","y","z"]),bounds.loc[[0]])
    trj = []

    ## The traps id are ordered (thankfully) in the same order as the particles, but they start consecutively.
    # We keep this order but start at one.
    traps.loc[:,"id"] = traps.index.get_level_values("id").values
    traps.loc[:,"frame"] = traps.index.get_level_values("frame")
    traps.loc[:,"id"] = traps["id"]-min(traps["id"])+1
    traps = traps.set_index(["frame","id"])

    ## create a relative position vector. This goes from the center of the trap to the position of the particle
    colloids = atoms-traps
    colloids = colloids[["x","y","z"]]
    colloids.columns = ["cx","cy","cz"]
    traps = pd.concat([traps,colloids],axis=1)
    colloids = []
    atoms = []

    ## Flip those traps that are not pointing in the  direction of the colloids
    flip = np.sign((traps[["dx","dy","dz"]].values*traps[["cx","cy","cz"]].values).sum(axis=1))
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*flip[:,np.newaxis]

    ## make the direction vector unitary
    mag = np.sign((traps[["dx","dy","dz"]].values**2).sum(axis=1))
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*mag[:,np.newaxis]

    #timestep = 10e-3 #sec
    #traps["t"] = traps.index.get_level_values("frame")*timestep

    return traps

def get_ice_trj_single(col,i):

    lz_rd = col.sim.lazy_read
    trj = lz_rd[slice(i,i+1)]
    trj["t"] = trj.index.get_level_values("frame")*col.sim.timestep.to("sec").magnitude
    bnd = lz_rd.get_bounds(slice(i,i+1))

    traps = trj[trj.type==2].copy(deep=True)
    traps = traps.rename(columns = {"mux":"dx","muy":"dy","muz":"dz"})
    atoms = trj[trj.type==1].copy(deep=True)
    moments = atoms.filter(["mux", "muy", "muz"])
    atoms = atoms.filter(["x","y","z"])

    traps.loc[:,"id"] = traps.index.get_level_values("id").values
    traps.loc[:,"frame"] = traps.index.get_level_values("frame")
    traps.loc[:,"id"] = traps["id"]-min(traps["id"])+1
    traps = traps.set_index(["frame","id"])

    colloids = atoms-traps
    colloids = colloids[["x","y","z"]]
    colloids.columns = ["cx","cy","cz"]
    traps = pd.concat([traps,colloids, moments],axis=1)

    for ax in ["x","y","z"]:
        unwrap_frame(traps,bnd,ax)

    ## Flip those traps that are not pointing in the  direction of the colloids
    flip = np.sign((traps[["dx","dy","dz"]].values*traps[["cx","cy","cz"]].values).sum(axis=1))
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*flip[:,np.newaxis]

    ## make the direction vector unitary
    mag = np.sign((traps[["dx","dy","dz"]].values**2).sum(axis=1))
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*mag[:,np.newaxis]
    return traps.drop(columns="type"), bnd

def get_ice_trj_low_memory_hdf(col):
    import tqdm.notebook as tqdm
    name = os.path.split(col.sim.base_name)[1]
    mode = "a"
    col.sim.load(read_trj=False)

    for i,t in tqdm.tqdm(enumerate(col.sim.lazy_read.T),
                                total = len(col.sim.lazy_read.T),
                                desc = "Iterating through file",
                                leave = False ):

        trj, bnd = get_ice_trj_single(col,i)

        trj.astype("float16").to_hdf(
            os.path.join(col.dir_name,name+".h5"), key = "trj",
            mode = mode,
            format = "table", append = True)

        bnd.astype("float16").to_hdf(
            os.path.join(col.dir_name,name+".h5"), key = "bounds",
            mode = mode,
            format = "table", append = True)

        mode = "a"

def get_ice_trj_low_memory(col):
    import tqdm.notebook as tqdm
    name = os.path.split(col.sim.base_name)[1]
    mode = "w"
    header = True
    col.sim.load(read_trj=False)

    for i,t in tqdm.tqdms(enumerate(col.sim.lazy_read.T),
                                total = len(col.sim.lazy_read.T),
                                desc = "Iterating through file" ):
        get_ice_trj_single(col,i)[0].to_csv(
            os.path.join(col.dir_name,name+".trj"), sep="\t",
            mode = mode, header = header)
        mode = "a"
        header = False
