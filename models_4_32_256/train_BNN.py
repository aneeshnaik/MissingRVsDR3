#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train BNN on Gaia DR3.

Created: August 2022
Author: A. P. Naik
"""
import os
import sys
import pandas as pd
import numpy as np
from batchgauss import sample
from banyan import BNN, BLoss

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


sys.path.append("..")
import src.utils as u
import src.params as p
from src.coords import convert_pos


def print_rank_message(rank, msg):
    print(f"(GPU {rank}) " + msg, flush=True)
    return


def get_obs_cov(df):

    # number of stars
    N = len(df)

    # obs array
    obs = np.stack(
        [df['ra'], df['dec'],
         df['pmra'], df['pmdec'], df['v_los']],
        axis=-1
    )

    # standard deviations
    s0 = (df['ra_err']).to_numpy()
    s1 = (df['dec_err']).to_numpy()
    s2 = (df['pmra_err']).to_numpy()
    s3 = (df['pmdec_err']).to_numpy()
    s4 = (df['v_los_err']).to_numpy()
    s_list = [s0, s1, s2, s3, s4]

    # construct correlation matrix
    r = np.zeros((N, 5, 5))
    for i in range(5):
        r[:, i, i] = 1
    r[:, 0, 1] = r[:, 1, 0] = (df['ra_dec_corr']).to_numpy()
    r[:, 0, 2] = r[:, 2, 0] = (df['ra_pmra_corr']).to_numpy()
    r[:, 0, 3] = r[:, 3, 0] = (df['ra_pmdec_corr']).to_numpy()
    r[:, 1, 2] = r[:, 2, 1] = (df['dec_pmra_corr']).to_numpy()
    r[:, 1, 3] = r[:, 3, 1] = (df['dec_pmdec_corr']).to_numpy()
    r[:, 2, 3] = r[:, 3, 2] = (df['pmra_pmdec_corr']).to_numpy()

    # covariance matrix
    cov = np.zeros_like(r)
    for i in range(5):
        for j in range(5):
            si = s_list[i]
            sj = s_list[j]
            cov[:, i, j] = r[:, i, j] * si * sj

    return obs, cov


def get_loader(d_matrix, obs, cov, rng, N_batch, x_mu, x_sig, y_mu, y_sig):

    # sample distance from distance matrix
    N = len(d_matrix)
    q = d_matrix[np.arange(N), rng.integers(low=0, high=10, size=N)]

    # sample ra/dec/pmra/pmdec/v_los from covariance matrix, stack with d
    q = np.hstack((q[:, None], sample(means=obs, covs=cov, rng=rng)))

    # convert ra/dec/d to X/Y/Z (note d = 1/parallax)
    X, Y, Z = convert_pos(q[:, 1], q[:, 2], q[:, 0]).T

    # create torch tensors
    x = torch.tensor(np.stack([X, Y, Z, q[:, 3], q[:, 4]], axis=-1)).double()
    y = torch.tensor(q[:, 5])[:, None].double()

    # rescale units
    x = u.rescale_data(x, mu=x_mu, sig=x_sig)
    y = u.rescale_data(y, mu=y_mu, sig=y_sig)

    # construct torch loader
    dset = TensorDataset(x, y)
    sampler = DistributedSampler(dset, shuffle=True, drop_last=True)
    loader = DataLoader(dset, batch_size=N_batch, sampler=sampler)
    return loader


def run_process(rank, N_gpu, seed, d_matrix, obs, cov):

    # initialize the process group
    print_rank_message(rank, "Initialising process group:")
    dist.init_process_group("nccl", rank=rank, world_size=N_gpu)
    print_rank_message(rank, ">>>Done.")

    # initialize RNG
    print_rank_message(rank, "Initialising RNG:")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    print_rank_message(rank, ">>>Done.")

    # construct model
    print_rank_message(rank, "Constructing model:")
    model = BNN(N_in=5, N_out=1, N_hidden=p.N_hidden, N_units=p.N_units)
    model.to(rank)
    model.double()
    model = DDP(model, device_ids=[rank])
    print_rank_message(rank, ">>>Done.")

    # construct other training objects
    print_rank_message(rank, "Constructing loss, optimiser and scheduler:")
    lossfn = BLoss()
    optim = Adam(model.parameters(), lr=p.lr0)
    scheduler = ReduceLR(optim, factor=p.lr_fac, min_lr=p.min_lr,
                         threshold=p.threshold, cooldown=p.cooldown)
    print_rank_message(rank, ">>>Done.")

    # loader args
    largs = dict(
        x_mu=p.x_mu, x_sig=p.x_sig, y_mu=p.y_mu, y_sig=p.y_sig,
        N_batch=p.N_batch, rng=rng
    )

    # on device 0, array for storing training data
    if rank == 0:
        training_data = []

    print_rank_message(rank, "Beginning training loop:")
    for epoch in range(p.N_epochs_max):

        # start timer on device 0
        if rank == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        loader = get_loader(d_matrix, obs, cov, **largs)
        loader.sampler.set_epoch(epoch)

        # loop over batches
        losses = []
        for i, (x_batch, y_batch) in enumerate(loader):

            # zero grad
            model.zero_grad(set_to_none=True)

            # forward pass
            preds = model(x_batch, p.N_samples)
            loss = lossfn(preds, y_batch.to(rank))
            losses.append(loss.item())

            # backward pass
            loss.backward()
            optim.step()

        # average loss, reduce across GPUs, step scheduler
        avg_loss = torch.tensor(losses).mean().to(rank)
        dist.barrier()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss.item()
        scheduler.step(avg_loss)

        # stop timer on device 0
        if rank == 0:
            end.record()
            torch.cuda.synchronize()
            t = start.elapsed_time(end)

        # on GPU 0, print message and store training data
        lr = optim.param_groups[0]['lr']
        if rank == 0:
            print(
                f"Epoch {epoch + 1} complete. "
                f"Loss: {avg_loss:.4f}, "
                f"LR: {lr:.3e}, "
                f"Num. bad epochs: {scheduler.num_bad_epochs}, "
                f"Total time: {t/1000}",
                flush=True
            )
            row = {
                'loss': avg_loss,
                'lr': lr,
                't': t / 1000
            }
            training_data.append(row)

        # save intermediate model every 4 epochs
        if (rank == 0) and ((epoch + 1) % 4 == 0):
            model.module.save(f'intermediate_models/{seed}_{epoch + 1}.pth')

        # end early if min lr reached
        if lr <= p.min_lr:
            break

    # clean up
    print_rank_message(rank, "Cleaning up process group:")
    dist.barrier()
    dist.destroy_process_group()
    print_rank_message(rank, ">>>Done.")

    # save
    if rank == 0:
        print("\nTraining complete. Saving model:", flush=True)
        model.module.save(f'{seed}.pth')
        df = pd.DataFrame(training_data)
        df.to_csv(f'{seed}_training_data.csv')
        print(">>>Done.", flush=True)

    return


if __name__ == "__main__":

    # device count
    N_gpu = torch.cuda.device_count()
    assert N_gpu > 0, "No GPU found."

    # script arg is random seed
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # start message
    print("\n", flush=True)
    print("%%%%%%%%%%%%%%%%", flush=True)
    print(f"Training a BNN with seed {seed} on {N_gpu} devices", flush=True)
    print("%%%%%%%%%%%%%%%%\n", flush=True)

    # set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # load data
    ddir = u.get_datadir()
    print(f"Found data directory: {ddir} || Loading data:", flush=True)
    df = pd.read_hdf(ddir + "train.hdf5")
    print(">>>Done.\n", flush=True)

    # construct distance matrix
    print("Constructing distance matrix:", flush=True)
    d_matrix = np.array([df[f'd{i}'] for i in range(10)]).T
    print(">>>Done.\n", flush=True)

    # construct covariance matrix
    print("Constructing covariance matrices:", flush=True)
    obs, cov = get_obs_cov(df)
    print(">>>Done.\n", flush=True)

    # spawn parallel jobs and run
    args = (N_gpu, seed, d_matrix, obs, cov)
    mp.spawn(run_process, args=args, nprocs=N_gpu, join=True)
