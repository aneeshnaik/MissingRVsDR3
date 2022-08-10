#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Match entries from EDR3-derived to prediction catalogue to DR3 stars.

Created: July 2022
Author: A. P. Naik
"""
import numpy as np
import pandas as pd
import sys
from h5py import File as hFile
from tqdm import trange

sys.path.append("..")
from src.utils import get_datadir, batch_calculate as batch


if __name__ == "__main__":

    print("Matching predictions with DR3 RVs:")

    # filepaths
    ddir = get_datadir()
    pred_cat = ddir + "EDR3_predictions/EDR3MissingRVCatalogue.hdf5"
    DR3_cat = ddir + "DR3_RVs.csv"
    savefile = ddir + "EDR3_predictions/EDR3_prediction_results.npz"
    auxfile = ddir + "EDR3_predictions/EDR3_prediction_aux.npz"

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

    # get true vels matching up ids of predicted vels
    print(">>>Match IDs")
    x = df['source_id'].to_numpy()
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], ids)
    indices = xsorted[ypos]
    v_true = df['radial_velocity'][indices].to_numpy()
    v_err = df['radial_velocity_error'][indices].to_numpy()
    G = df['phot_g_mean_mag'][indices].to_numpy()
    col = df['bp_rp'][indices].to_numpy()
    G_RVS = df['grvs_mag'][indices].to_numpy()
    T_eff = df['rv_template_teff'][indices].to_numpy()

    # calculate means and quantiles
    print(">>>Calculating means and quantiles")
    N = 500000
    mu = batch(v_pred, N, fn=np.mean, fn_args={'axis': -1})
    q16 = batch(v_pred, N, fn=np.percentile, fn_args={'axis': -1, 'q': 16})
    q84 = batch(v_pred, N, fn=np.percentile, fn_args={'axis': -1, 'q': 84})
    sig = (q84 - q16) / 2

    # save
    print(">>>Saving")
    np.savez(savefile, v_true=v_true, mu=mu, sig=sig)
    np.savez(auxfile, v_err=v_err, G=G, col=col, G_RVS=G_RVS, T_eff=T_eff)
    print(">>>Done.\n")
