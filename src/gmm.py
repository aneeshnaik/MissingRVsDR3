#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions relating to Gaussian Mixture Models (for fitting posteriors).

Created: May 2023
Author: A. P. Naik
"""
import numpy as np
from sklearn import mixture
from scipy import stats


def fit_1D_gmm(x, N_mix, N_init=2):
    """Fit 1D GMM with N_mix components to sample x (1D numpy array)."""

    # set up GMM
    gmm = mixture.GaussianMixture(n_components=N_mix, n_init=N_init)

    # fit
    gmm = gmm.fit(x[:, None])

    # get params
    weights = gmm.weights_
    means = gmm.means_.squeeze()
    variances = gmm.covariances_.squeeze()

    return weights, means, variances


def gmm_pdf(x, weights, means, variances):
    """PDF of Gaussian mixture, evaluated at points x."""

    # infer no. of components
    N_mix = len(weights)

    # loop over components, sum PDF
    pdf = 0
    for i in range(N_mix):
        loc = means[i]
        scale = np.sqrt(variances[i])
        pdf += weights[i] * stats.norm.pdf(x, loc=loc, scale=scale)

    return pdf


def gmm_cdf(x, weights, means, variances):
    """
    CDF of Gaussian mixture, evaluated at points x.
    x is 1D array, shape (N_pts), weights/means/variances all 1D arrays shape
    (N_mix).
    """

    # infer no. of components
    N_mix = len(weights)

    # loop over components, sum CDF
    cdf = 0
    for i in range(N_mix):
        loc = means[i]
        scale = np.sqrt(variances[i])
        cdf += weights[i] * stats.norm.cdf(x, loc=loc, scale=scale)

    return cdf


def gmm_cdf_batch(x, weights, means, variances):
    """
    CDF of Gaussian mixture, evaluated at points x, different GMM at each
    point.
    x is 1D array, shape (N_pts), weights/means/variances all 2D arrays shape
    (N_pts, N_mix).
    """

    # infer no. of components
    N_mix = weights.shape[1]

    # loop over components, sum CDF
    cdf = 0
    for i in range(N_mix):
        loc = means[:, i]
        scale = np.sqrt(variances[:, i])
        cdf += weights[:, i] * stats.norm.cdf(x, loc=loc, scale=scale)

    return cdf

def gmm_percentile(weights, means, variances, q, N_pts=10000):
    """Interpolate percentile q of GMM with given weights, means, variances."""

    # convert args to numpy arrays
    weights = np.array(weights)
    means = np.array(means)
    variances = np.array(variances)

    # interpolation points
    x_min = min(means - 10 * np.sqrt(variances))
    x_max = max(means + 10 * np.sqrt(variances))
    x = np.linspace(x_min, x_max, N_pts)

    # calculate CDF
    F = gmm_cdf(x, weights, means, variances)

    # linear interpolation
    # if q array-like, loop over q and calculate all percentiles, return array
    # else calculate single percentile, return float
    if hasattr(q, "__len__"):
        xq = []
        q = np.array(q)
        for Fq in q / 100:
            i = np.where(F < Fq)[0][-1]
            x0 = x[i]
            x1 = x[i + 1]
            F0 = F[i]
            F1 = F[i + 1]
            xq.append(x0 + (x1 - x0) * (Fq - F0) / (F1 - F0))
        xq = np.array(xq)
    else:
        Fq = q / 100
        i = np.where(F < Fq)[0][-1]
        x0 = x[i]
        x1 = x[i + 1]
        F0 = F[i]
        F1 = F[i + 1]
        xq = x0 + (x1 - x0) * (Fq - F0) / (F1 - F0)

    return xq
