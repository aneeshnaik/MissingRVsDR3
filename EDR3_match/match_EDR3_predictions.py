#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Match entries from EDR3-derived prediction catalogue to DR3 stars. Saves match
HDF5 file in EDR3_predictions subdirectory of data directory.

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


def get_CDF(v_true, v_pred, h, batch_size=10000):

    y = np.zeros_like(v_true)
    N_samples = v_pred.shape[1]
    N = len(v_true)
    N_batches = N // batch_size

    # loop over batches
    for i in trange(N_batches):
        i0 = i * batch_size
        i1 = (i + 1) * batch_size
        x = 0.5 * (v_true[i0:i1, None] - v_pred[i0:i1]) / h[i0:i1, None]
        y[i0:i1] = np.sum(0.5 * (np.tanh(x) + 1), axis=-1) / N_samples

    # remainder data
    if N % batch_size != 0:
        i0 = N_batches * batch_size
        x = 0.5 * (v_true[i0:, None] - v_pred[i0:]) / h[i0:, None]
        y[i0:] = np.sum(0.5 * (np.tanh(x) + 1), axis=-1) / N_samples

    return y


if __name__ == "__main__":

    print("Matching predictions with DR3 RVs:")

    # filepaths
    ddir = get_datadir()
    pred_cat = ddir + "EDR3_predictions/EDR3MissingRVCatalogue.hdf5"
    DR3_cat = ddir + "DR3_6D/DR3_6D.csv"
    dist_cat = ddir + "DR3_6D/distance_samples.npy"
    savefile = ddir + "EDR3_predictions/EDR3_prediction_match.hdf5"

    # load predicted catalogue
    print(">>>Loading prediction catalogue")
    with hFile(pred_cat, 'r') as hf:
        ids = hf["ids"][:]
        v_pred = hf["v_samples"][:]

    # load DR3 RVs and distance samples
    print(">>>Loading observed DR3 RVs")
    cols = [
        'source_id',
        'phot_g_mean_mag',
        'rv_template_teff',
        'radial_velocity',
        'radial_velocity_error',
        'parallax',
        'parallax_error',
        'ra',
        'dec',
        'pmra',
        'pmdec',
    ]
    df = pd.read_csv(DR3_cat, usecols=cols)
    dists = np.load(dist_cat)

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

    # append distance mean and SD to df
    df['dist_mean'] = np.mean(dists[indices], axis=-1).astype(np.float32)
    df['dist_err'] = np.std(dists[indices], axis=-1).astype(np.float32)

    # calculate means and quantiles
    print(">>>Calculating means and quantiles")
    N = 500000
    mu = batch(v_pred, N, fn=np.mean, fn_args={'axis': -1})
    q16 = batch(v_pred, N, fn=np.percentile, fn_args={'axis': -1, 'q': 16})
    q84 = batch(v_pred, N, fn=np.percentile, fn_args={'axis': -1, 'q': 84})
    std = batch(v_pred, N, fn=np.std, fn_args={'axis': -1})
    sig = (q84 - q16) / 2

    # calculate posterior positions
    print(">>>Calculating F values")
    N_samples = v_pred.shape[1]
    h = 0.6 * std * np.power(N_samples, -0.2)
    F = get_CDF(df['radial_velocity'].to_numpy(), v_pred, h)

    # appending prediction mean/sig and F val to dataframe
    df['mu_pred'] = mu.astype(np.float32)
    df['sig_pred'] = sig.astype(np.float32)
    df['F'] = F.astype(np.float32)

    # single generation of velocity posterior
    df['single_prediction'] = v_pred[:, 0].astype(np.float32)

    # save
    print(">>>Saving")
    df.to_hdf(savefile, 'EDR3Match', index=False, mode='w')
    print(">>>Done.\n")
