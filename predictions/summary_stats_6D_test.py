#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Having generated predictions on 6D test set, calculate and save summary stats.

Created: April 2023
Author: A. P. Naik
"""
import numpy as np
import sys

sys.path.append("..")
from src.utils import get_datadir, batch_calculate


if __name__ == "__main__":

    # load data
    ddir = get_datadir()
    data = np.load(ddir + 'DR3_predictions/6D_test.npz')
    v_pred = data['v_los_preds']
    v_true = data['v_los_true']
    N_samples = v_pred.shape[1]

    # batch calculate statistics: 16/84 percentiles, mean, std
    q16 = batch_calculate(v_pred, 10000, np.percentile, {'q': 16, 'axis': -1})
    q84 = batch_calculate(v_pred, 10000, np.percentile, {'q': 84, 'axis': -1})
    mu = batch_calculate(v_pred, 10000, np.mean, {'axis': -1})
    std = batch_calculate(v_pred, 10000, np.std, {'axis': -1})

    # logistic kernel width
    h = 0.6 * std * np.power(N_samples, -0.2)

    # posterior 'width'
    sig = (q84 - q16) / 2

    # save
    savefile = ddir + 'DR3_predictions/6D_test_summary_stats.npz'
    np.savez(savefile, mu=mu, sig=sig, h=h)
