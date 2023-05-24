#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine individual GMM sub-catalogues for the 5D set.

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

    # parse argument
    assert len(sys.argv) == 2
    dset_ind = int(sys.argv[1])

    # directory/file names
    ddir = get_datadir()
    batchdir = ddir + f"DR3_predictions/5D_{dset_ind}_GMM_batches/"
    savefile = ddir + f"DR3_predictions/5D_{dset_ind}_GMM_catalogue.hdf5"
    savekey = f"GMMCatalogue_5D_{dset_ind}"

    # loop over batches
    print("Looping over GMM batches:", flush=True)
    dfs = []
    for i in trange(64):
        dfs.append(pd.read_hdf(batchdir + f"5D_{dset_ind}_GMM_batch_{i}.hdf5"))
    print(">>>Done.\n", flush=True)

    # concatenate full DataFrame
    print("Concatenating", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    print(">>>Done.\n", flush=True)

    # save
    print("Saving:", flush=True)
    df.to_hdf(savefile, savekey, index=False, mode='w')
    print(">>>Done.\n", flush=True)
