#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile catalogue of LAMOST RVs (w/ Gaia source ids) from combination of DR8
LRS AFGK catalogue and full MRS catalogue (bulk-downloaded catalogues). Apply
quality cuts and perform variance-weighted sum over multiple observations of
single object.

Created: May 2023
Author: A. P. Naik
"""
import sys
import pandas as pd
import numpy as np
from tqdm import trange

sys.path.append("..")
from src.utils import get_datadir


def load_catalogues():

    ddir = get_datadir()
    LRSfile = ddir + "external/LAMOST_DR8_LRS_stellar.csv"
    MRSfile = ddir + "external/LAMOST_DR8_MRS.csv"
    lrs_cols = [
        'gaia_source_id',
        'rv', 'rv_err',
        'snru', 'snrg', 'snrr', 'snri', 'snrz'
    ]
    mrs_cols = [
        'gaia_source_id',
        'snr',
        'rv_br1', 'rv_br1_err', 'rv_br_flag'
    ]
    lrs = pd.read_csv(LRSfile, usecols=lrs_cols)
    mrs = pd.read_csv(MRSfile, usecols=mrs_cols)

    return lrs, mrs


def apply_quality_cuts(lrs, mrs):
    m1 = ((lrs['snru'] > 5)
          & (lrs['snrg'] > 5)
          & (lrs['snrr'] > 5)
          & (lrs['snri'] > 5)
          & (lrs['snrz'] > 5)
          & (lrs['rv_err'] > 0)
          & (lrs['gaia_source_id'] != -9999))
    m2 = ((mrs['rv_br_flag'] == 0)
          & (mrs['snr'] > 5)
          & (mrs['gaia_source_id'] != -9999))
    return lrs[m1], mrs[m2]


def get_velocities_dataframe(lrs, mrs):

    # concatenate
    rv = np.hstack((lrs['rv'], mrs['rv_br1']))
    rv_err = np.hstack((lrs['rv_err'], mrs['rv_br1_err']))
    ids = np.hstack((lrs['gaia_source_id'], mrs['gaia_source_id']))

    # sort
    inds = np.argsort(ids)
    rv = rv[inds]
    rv_err = rv_err[inds]
    ids = ids[inds]

    # inverse variances (weights for v averaging)
    w = 1 / rv_err**2

    # get unique source_ids and counts
    uids, uid_counts = np.unique(ids, return_counts=True)

    # loop over unique source ids, perform weighted average
    v = np.zeros_like(uids, dtype=np.float32)
    v_err = np.zeros_like(uids, dtype=np.float32)
    counter = 0
    for i in trange(len(uids)):
        count = uid_counts[i]
        i0 = counter
        i1 = counter + count
        v[i] = np.average(rv[i0:i1], weights=w[i0:i1])
        v_err[i] = np.sqrt(1 / np.sum(w[i0:i1]))
        counter += count

    # construct DataFrame
    df = pd.DataFrame()
    df['gaia_source_id'] = uids
    df['v'] = v
    df['v_err'] = v_err

    return df


if __name__ == "__main__":

    # load bulk LAMOST catalogues (LR sample and MR sample)
    print("Loading catalogues:")
    lrs, mrs = load_catalogues()
    print(">>>Done.\n")

    # apply initial quality cuts
    print("Quality cuts:")
    lrs, mrs = apply_quality_cuts(lrs, mrs)
    print(">>>Done.\n")

    # compile averaged catalogue
    print("Computing velocities:")
    df = get_velocities_dataframe(lrs, mrs)
    print(">>>Done.\n")

    # additional v_err cut
    print("Error cut:")    
    df = df[df['v_err'] < 3.5]
    print(">>>Done.\n")

    # save
    print("Saving:")
    savefile = get_datadir() + "external/LAMOST_xmatch.hdf5"
    df.to_hdf(savefile, "LAMOST_xmatch", index=False, mode='w')
    print(">>>Done.\n")
