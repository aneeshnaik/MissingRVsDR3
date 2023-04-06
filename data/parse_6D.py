#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From raw Gaia .csv file, construct .hdf5 files for training/test 6D sets.

Created: August 2022
Author: A. P. Naik
"""
import numpy as np
import pandas as pd
from sys import path

path.append("..")
import src.utils as u
import src.params as p


if __name__ == '__main__':

    print("Parsing DR3 6D data:")

    # random number generator
    rng = np.random.default_rng(42)

    # data directory
    ddir = u.get_datadir()

    # columns to read
    cols = [
        'source_id',
        'radial_velocity', 'radial_velocity_error',
        'ra', 'dec', 'pmra', 'pmdec',
        'ra_error', 'ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
        'dec_error', 'dec_pmra_corr', 'dec_pmdec_corr',
        'pmra_error', 'pmra_pmdec_corr',
        'pmdec_error'
    ]

    # read DR3 file
    print(">>> Loading DR3 data")
    df = pd.read_csv(ddir + 'DR3_6D/DR3_6D.csv', usecols=cols)

    # load Rybizki flags
    print(">>> Loading astrometric fidelities")
    df1 = pd.read_csv(ddir + 'DR3_6D/rybizki_flags.csv')

    # merge
    print(">>> Merging fidelities")
    df = df.merge(df1, on='source_id')

    # read distances
    print(">>> Reading distances")
    d = np.load(ddir + 'distance_samples.npy')

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

    # train/test split
    print(">>> Performing train/test split")
    df_tr, df_te = u.train_test_split(df, 0.8, rng=rng)

    # construct distance matrix from training set
    print(">>> Constructing distance matrix")
    d_matrix = np.array([df_tr[f'd{i}'] for i in range(10)]).T
    mu_d = np.mean(d_matrix, axis=-1)
    sig_d = np.std(d_matrix, axis=-1)

    # quality cuts (on training set only)
    print(f">>> Quality cuts... pre-cut size {len(df_tr)}")
    m = ((df_tr['fidelity'] > 0.5)
         & (df_tr['v_los_err'] < p.VLOSERRCUT)
         & (df_tr['pmdec_err'] < p.PMERRCUT)
         & (df_tr['pmra_err'] < p.PMERRCUT)
         & (sig_d / mu_d < p.DERRCUT))
    df_tr = df_tr[m]
    print(f">>> Quality cuts... post-cut size {len(df_tr)}")

    # save
    print(">>> Saving")
    df_tr.to_hdf(f"{ddir}/DR3_6D/train.hdf5", "train", index=False, mode='w')
    df_te.to_hdf(f"{ddir}/DR3_6D/test.hdf5", "test", index=False, mode='w')
    print(">>> Done.")
