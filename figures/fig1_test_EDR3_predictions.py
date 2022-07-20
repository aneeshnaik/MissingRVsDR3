#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import pandas as pd
import numpy as np
from os.path import exists
from h5py import File as hFile


def create_plot_data(dfile):
    
    # data directory
    ddir = 

    # load predicted catalogue
    with hFile("/Users/aneeshnaik/Science/Data/ProjectData/MissingRVsData/Predictions/EDR3MissingRVCatalogue.hdf5", 'r') as hf:
        ids = hf["ids"][:]
        v_pred = hf["v_samples"][:]

    # load DR3 RVs
    df = pd.read_csv("/Users/aneeshnaik/Science/Data/ProjectData/MissingRVsDR3Data/DR3_RVs.csv")

    # cut prediction catalogue down to stars in DR3
    m = np.isin(ids, df['source_id'])
    ids = ids[m]
    v_pred = v_pred[m]

    # get true vels matching up ids of predicted vels
    x = df['source_id'].to_numpy()
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], ids)
    indices = xsorted[ypos]
    v_true = df['radial_velocity'][indices]
    
    return


if __name__ == "__main__":

    # load plot data (create if not present)
    dfile = "fig1_test_EDR3_predictions_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)