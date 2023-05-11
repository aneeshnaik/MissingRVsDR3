#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine individual GMM sub-catalogues for the 5D set.

Created: May 2023
Author: A. P. Naik
"""
import pandas as pd
import sys

sys.path.append("..")
from src.utils import get_datadir

if __name__ == "__main__":

    # parse argument
    assert len(sys.argv == 2)
    dset_ind = int(sys.argv[1])

    # directory/file names
    ddir = get_datadir()
    batchdir = ddir + f"DR3_predictions/5D_{dset_ind}_GMM_batches/"
    savefile = ddir + f"DR3_predictions/5D_{dset_ind}_GMM_catalogue.hdf5"
    savekey = f"GMMCatalogue_5D_{dset_ind}"

    # loop over batches
    dfs = []
    for i in range(64):
        dfs.append(pd.read_hdf(batchdir + f"5D_{dset_ind}_GMM_batch_{i}.hdf5"))

    # concatenate full DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # save
    df.to_hdf(savefile, savekey, index=False, mode='w')
