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
    N_batches = 64   # no. of batches
    N_mix = 4        # no. of GMM components
    N_init = 3       # no. of random initialisations for GMM fitting

    # parse script argument" dataset_ind * N_batches + batch_ind
    # so arg should span 0 to (N_batches * N_datasets - 1)
    assert len(sys.argv) == 2
    arg = int(sys.argv[1])
    print("Parsing argument:", flush=True)
    dataset_ind, batch_ind = parse_argument(arg, N_batches, N_datasets)
    print(f">>>Got arguments dataset_ind: {dataset_ind}, batch_ind: {batch_ind}", flush=True)
    print(">>>Done.\n", flush=True)

    # filenames
    ddir = get_datadir()
    predfile = ddir + f"DR3_predictions/5D_{dataset_ind}_raw_predictions.npy"
    idfile = ddir + f"DR3_5D/{dataset_ind}.hdf5"
    savefile = ddir + f"DR3_predictions/5D_{dataset_ind}_GMM_batches/5D_{dataset_ind}_GMM_batch_{batch_ind}.hdf5"
    pvalfile = ddir + f"DR3_predictions/5D_{dataset_ind}_GMM_batches/5D_{dataset_ind}_GMM_pvals_{batch_ind}.npy"

    # load prediction catalogue (file large, use memmap)
    print("Loading raw predictions:", flush=True)
    preds = np.load(predfile, mmap_mode='r')
    print(">>>Done.\n", flush=True)

    # get star source_ids
    print("Loading star IDs:", flush=True)
    ids = pd.read_hdf(idfile)['source_id'].to_numpy()
    print(">>>Done.\n", flush=True)

    # batch; final batch bigger
    print("Batching:", flush=True)
    N_stars = len(preds)
    batch_size = N_stars // N_batches
    i0 = batch_ind * batch_size
    if batch_ind == N_batches - 1:
        preds = preds[i0:]
        ids = ids[i0:]
    else:
        preds = preds[i0:i0 + batch_size]
        ids = ids[i0:i0 + batch_size]
    print(">>>Done.\n", flush=True)

    # smooth preds with logistic kernel
    print("Smoothing:", flush=True)
    N_samples = preds.shape[1]
    h = 0.6 * np.std(preds, axis=-1) * np.power(N_samples, -0.2)
    preds = preds + h[:, None] * stats.logistic.rvs(size=N_samples)[None]
    print(">>>Done.\n", flush=True)

    # calculate sample mean and SD
    print("Calculating sample mean and SD:", flush=True)
    mu = np.mean(preds, axis=-1)
    std = np.std(preds, axis=-1)
    print(">>>Done.\n", flush=True)

    # loop over stars:
    #   - fit predictions w/ GMM
    #   - get GMM quantiles
    #   - KS test, get p value
    print("Starting GMM loop:", flush=True)
    weights = np.zeros((len(preds), N_mix))
    means = np.zeros_like(weights)
    vars = np.zeros_like(weights)
    percentiles = np.zeros((len(preds), 5))
    pvals = np.zeros(len(preds))
    for i in trange(len(preds)):

        # fit GMM
        w, mu, var = fit_1D_gmm(preds[i], N_mix=N_mix, N_init=N_init)

        # quantiles
        q = gmm_percentile(w, mu, var, q=(5, 15.9, 50, 84.1, 95))

        # KS-test
        p = stats.kstest(preds[i], gmm_cdf, args=(w, mu, var)).pvalue

        # put values in array
        weights[i] = w
        means[i] = mu
        vars[i] = var
        percentiles[i] = q
        pvals[i] = p
    print(">>>Done.\n", flush=True)

    # save catalogue and save p-values separately
    print("Saving:", flush=True)
    df = pd.DataFrame()
    df['source_id'] = ids
    df['sample_mean'] = mu.astype(np.float32)
    df['sample_std'] = std.astype(np.float32)
    df['q050'] = percentiles[:, 0].astype(np.float32)
    df['q159'] = percentiles[:, 1].astype(np.float32)
    df['q500'] = percentiles[:, 2].astype(np.float32)
    df['q841'] = percentiles[:, 3].astype(np.float32)
    df['q950'] = percentiles[:, 4].astype(np.float32)
    for i in range(N_mix):
        df[f'w{i}'] = weights[:, i].astype(np.float32)
    for i in range(N_mix):
        df[f'mu{i}'] = means[:, i].astype(np.float32)
    for i in range(N_mix):
        df[f'var{i}'] = vars[:, i].astype(np.float32)
    df.to_hdf(savefile, f"GMM_batch_{batch_ind}", index=False, mode='w')
    np.save(pvalfile, pvals)
    print(">>>Done.\n", flush=True)
