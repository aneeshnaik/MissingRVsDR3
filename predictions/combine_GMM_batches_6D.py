#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine individual GMM sub-catalogues for the 6D test set.

Created: May 2023
Author: A. P. Naik
"""
import pandas as pd
import sys

sys.path.append("..")
from src.utils import get_datadir

if __name__ == "__main__":

    # directory/file names
    ddir = get_datadir()
    batchdir = ddir + "DR3_predictions/6D_test_GMM_batches/"
    savefile = ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5"
    savekey = "GMMCatalogue_6D_test"

    # loop over batches
    dfs = []
    for i in range(64):
        dfs.append(pd.read_hdf(batchdir + f"6D_test_GMM_batch_{i}.hdf5"))

    # concatenate full DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # save
    df.to_hdf(savefile, savekey, index=False, mode='w')
