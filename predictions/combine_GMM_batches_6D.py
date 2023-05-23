#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine individual GMM sub-catalogues for the 6D test set.

Created: May 2023
Author: A. P. Naik
"""
import numpy as np
import pandas as pd
import sys
from tqdm import trange

sys.path.append("..")
from src.utils import get_datadir

if __name__ == "__main__":

    # directory/file names
    ddir = get_datadir()
    batchdir = ddir + "DR3_predictions/6D_test_GMM_batches/"
    savefile = ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5"
    predfile = ddir + "DR3_predictions/6D_test_raw_predictions.npy"
    savekey = "GMMCatalogue_6D_test"

    # loop over batches
    print("Looping over GMM batches:", flush=True)
    dfs = []
    for i in trange(64):
        dfs.append(pd.read_hdf(batchdir + f"6D_test_GMM_batch_{i}.hdf5"))
    print(">>>Done.\n", flush=True)

    # concatenate full DataFrame
    print("Concatenating", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    print(">>>Done.\n", flush=True)

    # change percentile data types
    print("Changing percentile dtypes:", flush=True)
    f = np.float32
    df = df.astype({'q159': f, 'q500': f, 'q841': f})
    print(">>>Done.\n", flush=True)

    # load prediction catalogue (file large, use memmap)
    print("Loading prediction catalogue:", flush=True)
    preds = np.load(predfile, mmap_mode='r')
    print(">>>Done.\n", flush=True)

    # get sample mu and sigma
    print("Calculating prediction means and standard deviations:", flush=True)
    df['sample_mean'] = np.mean(preds, axis=-1).astype(f)
    df['sample_std'] = np.std(preds, axis=-1).astype(f)
    print(">>>Done.\n", flush=True)

    # save
    print("Saving:", flush=True)
    df.to_hdf(savefile, savekey, index=False, mode='w')
    print(">>>Done.\n", flush=True)