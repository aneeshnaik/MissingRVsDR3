#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Match entries from EDR3-derived to prediction catalogue to DR3 stars. Saves
two files in EDR3_predictions subdirectory of data directory, both numpy
archive files (saved with np.savez, can be opened with np.load):
    - EDR3_prediction_results.npz: truths and mean/variation of predictions
    - EDR3_prediction_aux.hdf5: everything else for matched stars

Created: July 2022
Author: A. P. Naik
"""
import numpy as np
import pandas as pd
import sys
from h5py import File as hFile

sys.path.append("..")
from src.utils import get_datadir, batch_calculate as batch


if __name__ == "__main__":

    print("Matching predictions with DR3 RVs:")

    # filepaths
    ddir = get_datadir()
    pred_cat = ddir + "EDR3_predictions/EDR3MissingRVCatalogue.hdf5"
    DR3_cat = ddir + "DR3_6D/DR3_6D.csv"
    savefile = ddir + "EDR3_predictions/EDR3_prediction_results.npz"
    auxfile = ddir + "EDR3_predictions/EDR3_prediction_aux.hdf5"

    # load predicted catalogue
    print(">>>Loading prediction catalogue")
    with hFile(pred_cat, 'r') as hf:
        ids = hf["ids"][:]
        v_pred = hf["v_samples"][:]

    # load DR3 RVs
    print(">>>Loading observed DR3 RVs")
    df = pd.read_csv(DR3_cat)

    # cut prediction catalogue down to stars in DR3
    print(">>>Cutting prediction catalogue down to stars in DR3")
    m = np.isin(ids, df['source_id'])
    ids = ids[m]
    v_pred = v_pred[m]

    # cut DR3 down to match
    print(">>>Match IDs")
    x = df['source_id'].to_numpy()
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], ids)
    indices = xsorted[ypos]
    df = df.loc[indices]

    # calculate means and quantiles
    print(">>>Calculating means and quantiles")
    N = 500000
    mu = batch(v_pred, N, fn=np.mean, fn_args={'axis': -1})
    q16 = batch(v_pred, N, fn=np.percentile, fn_args={'axis': -1, 'q': 16})
    q84 = batch(v_pred, N, fn=np.percentile, fn_args={'axis': -1, 'q': 84})
    sig = (q84 - q16) / 2

    # save
    print(">>>Saving")
    np.savez(savefile, v_true=df['radial_velocity'].to_numpy(), mu=mu, sig=sig)
    df.to_hdf(auxfile, 'EDR3aux')
    print(">>>Done.\n")
