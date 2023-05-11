#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Take predictions for 5D set, fit with GMMs, save catalogue.

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


def parse_argument(arg, N_batches, N_datasets):
    dataset_ind = arg // N_batches
    batch_ind = arg % N_batches
    assert dataset_ind <= N_datasets - 1
    return dataset_ind, batch_ind


if __name__ == "__main__":

    # parameters
    N_datasets = 32  # no. of datasets
    N_batches = 64  # no. of batches
    N_mix = 4       # no. of GMM components
    N_init = 3      # no. of random initialisations for GMM fitting

    # parse script argument" dataset_ind * N_batches + batch_ind
    # so arg should span 0 to (N_batches * N_datasets - 1)
    assert len(sys.argv) == 2
    arg = int(sys.argv[1])
    dataset_ind, batch_ind = parse_argument(arg, N_batches, N_datasets)

    # filenames
    ddir = get_datadir()
    predfile = ddir + f"DR3_predictions/5D_{dataset_ind}_raw_predictions.npy"
    idfile = ddir + f"DR3_5D/{dataset_ind}.hdf5"
    savefile = ddir + f"DR3_predictions/5D_{dataset_ind}_GMM_batches/5D_{dataset_ind}_GMM_batch_{batch_ind}.hdf5"
    pvalfile = ddir + f"DR3_predictions/5D_{dataset_ind}_GMM_batches/5D_{dataset_ind}_GMM_pvals_{batch_ind}.npy"

    # load prediction catalogue (file large, use memmap)
    preds = np.load(predfile, mmap_mode='r')

    # get star source_ids
    ids = pd.read_hdf(idfile)['source_id'].to_numpy()

    # batch; final batch smaller
    N_stars = len(preds)
    batch_size = N_stars // N_batches
    i0 = batch_ind * batch_size
    if batch_ind == N_batches - 1:
        preds = preds[i0:]
        ids = ids[i0:]
    else:
        preds = preds[i0:i0 + batch_size]
        ids = ids[i0:i0 + batch_size]

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

    # save catalogue and save p-values separately
    df = pd.DataFrame()
    df['source_id'] = ids
    df['q159'] = percentiles[:, 0]
    df['q500'] = percentiles[:, 1]
    df['q841'] = percentiles[:, 2]
    for i in range(N_mix):
        df[f'w{i}'] = weights[:, i].astype(np.float32)
    for i in range(N_mix):
        df[f'mu{i}'] = means[:, i].astype(np.float32)
    for i in range(N_mix):
        df[f'var{i}'] = vars[:, i].astype(np.float32)
    df.to_hdf(savefile, f"GMM_batch_{batch_ind}", index=False, mode='w')
    np.save(pvalfile, pvals)
