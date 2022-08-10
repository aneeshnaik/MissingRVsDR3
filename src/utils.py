#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions.

Created: June 2022
Author: A. P. Naik
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


def get_datadir():
    ddir = os.environ['MRVDR3DDIR']
    assert os.path.exists(ddir)
    if ddir[-1] != '/':
        ddir += '/'
    return ddir


def rescale_data(data, mu, sig):
    return (data - mu) / sig


def construct_data_loader(x, y, N_batch, shuffle=True, drop_last=True):
    dset = TensorDataset(x, y)
    loader = DataLoader(
        dset, batch_size=N_batch, shuffle=shuffle, drop_last=drop_last,
        pin_memory=True
    )
    return loader


def find_torch_device(device, verbose=False):

    if device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            d_name = torch.cuda.get_device_name(device)
            if verbose:
                print(f"\nCUDA available. Using device '{d_name}'.\n", flush=True)
        else:
            if verbose:
                print("CUDA unavailable / GPU not found. Using CPU.\n", flush=True)
            device = torch.device("cpu")
    elif device == 'cpu':
        device = torch.device("cpu")
    else:
        raise ValueError("Device not recognised, should be 'cpu' or 'gpu'.")

    return device


def subsample(input, N_sample, rng=None, return_both=False):
    if rng is None:
        rng = np.random.default_rng(42)

    inds = rng.choice(np.arange(len(input)), size=N_sample, replace=False)
    mask = np.isin(np.arange(len(input)), inds)

    if return_both:
        return input[mask], input[~mask]

    return input[mask]


def train_test_split(input, train_frac, rng=None):
    N_train = int(train_frac * len(input))
    train, test = subsample(input, N_train, rng=rng, return_both=True)
    return train, test


def batch_calculate(data, batch_size, fn, fn_args):

    N = len(data)
    N_batches = N // batch_size

    # loop over batches
    q = np.array([], dtype=data.dtype)
    for i in trange(N_batches):
        i0 = i * batch_size
        i1 = (i + 1) * batch_size
        q = np.append(q, fn(data[i0:i1], **fn_args))

    # remainder data
    if N % batch_size != 0:
        i0 = N_batches * batch_size
        q = np.append(q, fn(data[i0:], **fn_args))

    return q
