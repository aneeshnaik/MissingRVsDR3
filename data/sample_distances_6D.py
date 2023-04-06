#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample Bailer-Jones geometric distances for RV stars.

Created: September 2022
Author: A. P. Naik
"""
import numpy as np
import pandas as pd
import warnings
import sys
from tqdm import tqdm
from zero_point import zpt
from scipy.special import gamma
from invsample import sample_family as sample

sys.path.append('..')
from src.utils import get_datadir


def prior(x, alpha, beta, L):
    """Generalised gamma distribution prior."""
    t1 = alpha / (gamma((beta + 1) / alpha) * L**(beta + 1))
    t2 = x**beta
    t3 = np.exp(-(x / L)**alpha)
    p = t1 * t2 * t3
    return p


def likelihood(x, w, wzp, sig_w):
    """Gaussian likelihood."""
    t1 = np.exp(-0.5 * (w - wzp - (1 / x))**2 / sig_w**2)
    t2 = 1 / (np.sqrt(2 * np.pi) * sig_w)
    p = t1 * t2
    return p


def posterior(q, alpha, beta, L, w, wzp, sig_w):
    """Posterior = prior * likelihood. Coordinate q=ln(distance)."""
    # calculate prior and likelihood
    x = np.exp(q)
    p_pr = prior(x, alpha, beta, L)
    p_li = likelihood(x, w, wzp, sig_w)

    # if star has zero likelihood everywhere, set to 1 (just sample prior)
    p_li[np.all(p_li == 0, axis=-1)] = 1

    # posterior; times x bc posterior in ln(x)
    p = x * p_pr * p_li
    return p


if __name__ == "__main__":

    # some numbers
    N_tot = 33653049                     # total number of 6D stars
    N_samples = 10                       # number of distance samples per star
    chunksize = 1000                     # chunksize for data read
    N_chunks = (N_tot // chunksize) + 1  # number of chunks
    N_filled = 0                         # current number of stars completed

    # array that will be filled with distance samples
    samples = np.zeros((N_tot, N_samples))

    # load Gaia data as pandas DataFrame
    ddir = get_datadir()
    df_iter = pd.read_csv(ddir + "/DR3_6D/DR3_6D.csv", chunksize=chunksize)

    # loop over chunks
    for i, df in tqdm(enumerate(df_iter), total=N_chunks):

        # get parallax zero-point for all stars in chunk
        # ignore warning re: pseudocolour extrapolation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zpt.load_tables()
            wzp = zpt.get_zpt(
                df['phot_g_mean_mag'],
                df['nu_eff_used_in_astrometry'],
                df['pseudocolour'],
                df['ecl_lat'],
                df['astrometric_params_solved']
            )
            wzp[~np.isfinite(wzp)] = -0.017

        # healpix numbers -> GGD prior params alpha, beta, L
        hpix = np.floor(df['source_id'] / 562949953421312).astype(np.int32)
        prior_data = pd.read_csv(ddir + '/DR3_6D/BJ_distances_prior.csv')
        alpha = prior_data.loc[hpix]['GGDalpha'].to_numpy()
        beta = prior_data.loc[hpix]['GGDbeta'].to_numpy()
        L = prior_data.loc[hpix]['GGDrlen'].to_numpy() / 1000

        # x_min and x_max for PDF->CDF integration. zoom in for high SNRs
        w = df['parallax'].to_numpy()
        sig_w = df['parallax_error'].to_numpy()
        SNR = (w - wzp) / sig_w
        x_min = np.log(0.0005) * np.ones(len(df))
        x_max = np.log(500) * np.ones(len(df))
        x_min[SNR > 50] = np.log(1 / (w - wzp - 5 * sig_w)[SNR > 50])
        x_max[SNR > 50] = np.log(1 / (w - wzp + 5 * sig_w)[SNR > 50])

        # roll together posterior arguments
        sample_args = {
            'prob_fn': posterior,
            'N_dists': len(df),
            'x_min': x_min,
            'x_max': x_max,
            'args': [
                alpha,
                beta,
                L,
                w,
                wzp,
                sig_w
            ]
        }

        # get samples; if 3000 too few points for CDF integration try 15000
        try:
            q_samples = sample(N=N_samples, N_pts=3000, **sample_args)
        except AssertionError:
            q_samples = sample(N=N_samples, N_pts=15000, **sample_args)
        x_samples = np.exp(q_samples)

        # feed into array
        samples[N_filled:N_filled + len(df)] = x_samples
        N_filled += len(df)

    # save
    np.save(ddir + '/DR3_6D/distance_samples', samples)
