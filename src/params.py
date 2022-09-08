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

# BNN hyperparameters
N_hidden = 8     # no. hidden layers
N_units = 64     # no. units per hidden layer
N_samples = 250  # no. samples

# training params
lr0 = 0.01
min_lr = 1e-5
lr_fac = 0.5
N_epochs_max = 500
N_batch = 6000

# unit rescalings
x_mu = torch.tensor([-8.0, 0, 0, 0, 0])
x_sig = torch.tensor([1.5, 1.5, 0.6, 15, 15])
y_mu = torch.tensor([0])
y_sig = torch.tensor([40])
