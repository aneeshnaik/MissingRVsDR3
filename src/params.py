#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various global parameter settings.

Created: August 2022
Author: A. P. Naik
"""
import torch

# quality cuts on training dataset
VLOSERRCUT = 8.5           # v_los_err < 8.5 km/s
PMERRCUT = 0.07            # mu_err < 0.07 mas/yr
DERRCUT = 0.05             # sig_d / mu_d < 0.05

# BNN hyperparameters
N_hidden = 4     # no. hidden layers
N_units = 32     # no. units per hidden layer
N_samples = 256  # no. samples

# training params
lr0 = 2e-4
min_lr = 5e-6
lr_fac = 0.5
cooldown = 5
threshold = 1e-4
N_epochs_max = 500
N_batch = 400

# unit rescalings
x_mu = torch.tensor([-8.0, 0, 0, 0, 0])
x_sig = torch.tensor([1.5, 1.5, 0.6, 15, 15])
y_mu = torch.tensor([0])
y_sig = torch.tensor([40])
