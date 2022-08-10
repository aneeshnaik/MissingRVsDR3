#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions relating to the model training.

Created: May 2022
Author: A. P. Naik
"""
import numpy as np


def train_epoch(model, device, loader, optim, scheduler, N_samples):

    # loop over train batches
    losses = np.array([])
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        loss = model.calc_loss(x, y, N_samples=N_samples)
        losses = np.append(losses, loss.item())
        loss.backward()
        optim.step()

    # step scheduler
    avg_loss = np.mean(losses)
    scheduler.step(avg_loss)
    lr = optim.param_groups[0]['lr']

    return avg_loss, lr
