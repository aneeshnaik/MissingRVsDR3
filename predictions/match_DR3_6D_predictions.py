#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Match DR3 6D predictions to truths. Saves file in DR3_predictions
subdirectory of data directory:
    - DR3_6D_stats.npz: truths and mean/variation of predictions

Created: July 2022
Author: A. P. Naik
"""
import pandas as pd
import numpy as np
import sys
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

    # load data
    print(">>>Loading predictions")
    ddir = get_datadir()
    data = np.load(ddir + 'DR3_predictions/6D_test.npz')
    v_pred = data['v_los_preds']
    v_true = data['v_los_true']
    N_samples = v_pred.shape[1]

    # calculate means and quantiles
    print(">>>Calculating means and quantiles")
    N = 500000
    mu = batch(v_pred, N, fn=np.mean, fn_args={'axis': -1})
    q16 = batch(v_pred, N, fn=np.percentile, fn_args={'axis': -1, 'q': 16})
    q84 = batch(v_pred, N, fn=np.percentile, fn_args={'axis': -1, 'q': 84})
    std = batch(v_pred, N, fn=np.std, fn_args={'axis': -1})
    sig = (q84 - q16) / 2

    # calculate F values
    print(">>>Calculating F values")
    N_samples = v_pred.shape[1]
    h = 0.6 * std * np.power(N_samples, -0.2)
    F = get_CDF(v_true, v_pred, h)

    # save
    print(">>>Saving")
    savefile = ddir + "DR3_predictions/DR3_6D_stats.npz"
    np.savez(savefile, v_true=v_true, mu=mu, sig=sig, F=F)
    print(">>>Done.\n")
