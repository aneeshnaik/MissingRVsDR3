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
        'radial_velocity', 'radial_velocity_error',
        'parallax',
        'ra',
        'dec',
        'pmra',
        'pmdec',
        'parallax_error',
        'ra_parallax_corr',
        'dec_parallax_corr',
        'parallax_pmra_corr',
        'parallax_pmdec_corr',
        'ra_error',
        'ra_dec_corr',
        'ra_pmra_corr',
        'ra_pmdec_corr',
        'dec_error',
        'dec_pmra_corr',
        'dec_pmdec_corr',
        'pmra_error',
        'pmra_pmdec_corr',
        'pmdec_error'
    ]

    # read file
    print(">>>Loading data")
    df = pd.read_csv(ddir + 'DR3_6D.csv', usecols=cols)

    # lose 2D stars GET RID OF THIS ONCE QUERY HAS CHANGED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(">>>Removing 2+1D stars")
    df = df.dropna()

    # lose -ve parallaxes GET RID OF THIS ONCE QUALITY FLAG INTRODUCED!!!!!!!!!!
    df = df[df['parallax'] > 0]

    # rename some columns
    print(">>>Renaming columns")
    df = df.rename(columns={"radial_velocity": "v_los",
                            "radial_velocity_error": "v_los_err",
                            "parallax_error": "parallax_err",
                            "ra_error": "ra_err",
                            "dec_error": "dec_err",
                            "pmra_error": "pmra_err",
                            "pmdec_error": "pmdec_err"})

    # shuffle
    print(">>>Shuffling DF")
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    # train/test split
    print(">>>Performing train/test split")
    df_tr, df_te = u.train_test_split(df, 0.9, rng=rng)

    # quality cuts (on training set only)
    print(">>>Quality cuts")
    f = df_tr['parallax_err'] / df_tr['parallax']
    m = ((f < p.PARALLAXFRACERRCUT)
         & (df_tr['v_los_err'] < p.VLOSERRCUT)
         & (df_tr['pmdec_err'] < p.PMERRCUT)
         & (df_tr['pmra_err'] < p.PMERRCUT))
    df_tr = df_tr[m]

    # save
    print(">>>Saving")
    df_tr.to_hdf(f"{ddir}train.hdf5", "train", index=False, mode='w')
    df_te.to_hdf(f"{ddir}test.hdf5", "test", index=False, mode='w')
    print(">>>Done.")
