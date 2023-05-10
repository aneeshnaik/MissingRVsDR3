#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Take predictions for 6D test set, fit with GMMs, save catalogue.

Created: May 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd

from tqdm import trange
from scipy import stats

sys.path.append("..")
from src.utils import get_datadir
from src.gmm import fit_1D_gmm, gmm_percentile, gmm_cdf


if __name__ == "__main__":

    # parameters
    N_mix = 4  # no. of GMM components
    N_init = 3  # no. of random initialisations for GMM fitting

    # filenames
    ddir = get_datadir()
    predfile = ddir + "DR3_predictions/6D_test_raw_predictions.npy"
    idfile = ddir + "DR3_6D/test.hdf5"
    savefile = ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5"
    pvalfile = ddir + "DR3_predictions/6D_test_pvals.npy"

    # load prediction catalogue (file large, use memmap)
    preds = np.load(predfile, mmap_mode='r')

    # loop over stars:
    #   - fit predictions w/ GMM
    #   - get GMM quantiles
    #   - KS test, get p value
    weights = np.zeros((len(preds), N_mix))
    means = np.zeros_like(weights)
    vars = np.zeros_like(weights)
    percentiles = np.zeros((len(preds), 3))
    pvals = np.zeros(len(preds))
    for i in trange(len(preds)):

        # fit GMM
        w, mu, var = fit_1D_gmm(preds[i], N_mix=N_mix, N_init=N_init)

        # quantiles
        q = gmm_percentile(w, mu, var, q=(15.9, 50, 84.1))

        # KS-test
        p = stats.kstest(preds[i], gmm_cdf, args=(w, mu, var)).pvalue

        # put values in array
        weights[i] = w
        means[i] = mu
        vars[i] = var
        percentiles[i] = q
        pvals[i] = p

    # get star source_ids
    ids = pd.read_hdf(idfile)['source_id']

    # save catalogue and save p-values separately
    df = pd.DataFrame()
    df['source_id'] = ids.to_numpy()
    df['q159'] = percentiles[:, 0]
    df['q500'] = percentiles[:, 1]
    df['q841'] = percentiles[:, 2]
    for i in range(N_mix):
        df[f'w{i}'] = weights[:, i].astype(np.float32)
    for i in range(N_mix):
        df[f'mu{i}'] = means[:, i].astype(np.float32)
    for i in range(N_mix):
        df[f'var{i}'] = vars[:, i].astype(np.float32)
    df.to_hdf(savefile, "gmm_catalogue", index=False, mode='w')
    np.save(pvalfile, pvals)
