#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From raw Gaia .csv file, construct .hdf5 files for training/test 6D sets.

Created: August 2022
Author: A. P. Naik
"""
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
import src.utils as u
import src.params as p


if __name__ == '__main__':

    # dataset index is script argument
    ind = sys.argv[1]

    # truncation DELETE THIS #####################################################
    nrows = 10000

    # print start message
    print("Parsing DR3 5D data:")

    # random number generator
    rng = np.random.default_rng(42)

    # data directory
    ddir = u.get_datadir()

    # columns to read
    cols = [
        'source_id',
        'ra', 'dec', 'pmra', 'pmdec',
        'ra_error', 'ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
        'dec_error', 'dec_pmra_corr', 'dec_pmdec_corr',
        'pmra_error', 'pmra_pmdec_corr',
        'pmdec_error'
    ]

    # load DR3 5D data
    print(">>> Loading DR3 data")
    df = pd.read_csv(ddir + f'DR3_5D/DR3_5D_{ind}.csv', usecols=cols)

    # read distances
    print(">>> Reading distances")
    d = np.load(ddir + f'DR3_5D/distance_samples_{ind}.npy')

    # add distances as columns
    print(">>> Appending distances to DataFrame")
    N_samples = d.shape[1]
    for i in range(N_samples):
        df[f'd{i}'] = d[:, i]

    # rename some columns
    print(">>> Renaming columns")
    df = df.rename(columns={"radial_velocity": "v_los",
                            "radial_velocity_error": "v_los_err",
                            "ra_error": "ra_err",
                            "dec_error": "dec_err",
                            "pmra_error": "pmra_err",
                            "pmdec_error": "pmdec_err"})

    # shuffle
    print(">>> Shuffling DF")
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    # save
    print(">>> Saving")
    df.to_hdf(ddir + f"DR3_5D/{ind}.hdf5", "train", index=False, mode='w')
    print(">>> Done.")
