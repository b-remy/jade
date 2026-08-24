import os
import argparse
import yaml
import math

import jax
import jax.numpy as jnp

from flax import nnx
import orbax.checkpoint as ocp

import matplotlib.pyplot as plt
import numpy as np

from jade.nn_hybrid import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.flow import Denoiser
from jade.utils import load_model
from jade.lensing import ks93inv

from jade.sampling import EulerSampler, HeunSampler
from train import plot_denoiser, normalize_batch

from datasets import load_from_disk

from functools import partial
from tqdm import tqdm

import yaml
import pickle

# LSST noise level
from jade.init import sigma_lsst

# Million-sample model (run fk49rnft), trained on sbi_lens_million_full.
CKPT_DIR = "/u/bremy/repos/jade/experiments/wandb/run-20260615_153845-fk49rnft/files/checkpoints"
DATASET = "/work/hdd/benb/bremy/sbi_lens_million_full"
OUT_DIR = "tarp_results/million"


def main():

    cfg, states = load_model(CKPT_DIR, "JADE_B_16_ema_best")

    model = JADE_B_16(
        rngs=nnx.Rngs(cfg['training']['seed']),
        in_channels=cfg['model']['in_channels'],
        input_size=cfg['model']['input_size'],
        enable_cond_image=cfg['model']['enable_cond_image'],
        cond_channels=cfg['model']['cond_channels'],
        num_cosmo_tokens=cfg['model']['num_cosmo_tokens'],
        cond_patch_size=cfg['model']['cond_patch_size'],
        cond_start=cfg['model']['cond_start'],
        attn_drop=cfg['model']['attn_drop'],
        proj_drop=cfg['model']['proj_drop'],
        # Stage-2 runs set this; older stage-1 configs don't have the key.
        split_qkv=cfg['model'].get('split_qkv', False),
        mask_theta_to_field=cfg['model'].get('mask_theta_to_field', False),
    )

    model = Denoiser(model, cfg)
    nnx.update(model, states)

    # Build the sampler once at module scope so it is traced/compiled exactly
    # once when posterior_sampling is first called — never per-condition or
    # per-loop-step.
    sampler = HeunSampler(model=model, num_steps=128)
    num_samples = 500

    @jax.jit
    def make_cond(x, key):
        cond = x + sigma_lsst * jax.random.normal(key, shape=x.shape)
        cond = (cond - FIELD_MEAN.reshape(1, 1, 1, -1)) / FIELD_STD.reshape(1, 1, 1, -1)
        return cond

    @jax.jit
    def posterior_sampling(cond_i, key):
        keys = jax.random.split(key, 3)

        x_0 = jax.random.normal(keys[0], shape=(num_samples, 128, 128, 5))
        cosmo_0 = jax.random.normal(keys[1], shape=(num_samples, 6))

        x_samples, cosmo_samples = jax.vmap(sampler, in_axes=(0, 0, None, 0))(
            x_0, cosmo_0, cond_i, jax.random.split(keys[2], num_samples)
        )
        x_samples = x_samples * FIELD_STD + FIELD_MEAN
        cosmo_samples = cosmo_samples * THETA_STD + THETA_MEAN
        return x_samples, cosmo_samples

    # Evaluate TARP on the held-out test split only, reproducing the exact
    # train/test split used during training of fk49rnft (train_sharding.py:
    # train_test_split(test_size=val_split, seed=shuffle_seed)). Iterating the
    # full dataset here would draw ~95% in-sample (training) observations and
    # make the coverage look artificially good.
    dataset = load_from_disk(DATASET)
    dataset = dataset.train_test_split(
        test_size=cfg['data']['val_split'],
        seed=cfg['data']['shuffle_seed'],
    )["test"]
    dataset = dataset.with_format("numpy")

    loader = dataset.iter(batch_size=100)

    job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    n_jobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 5))

    print(f"Starting job {job_id} out of {n_jobs}")

    key = jax.random.PRNGKey(0)
    data_batch = next(loader)
    for _ in range(job_id):
        data_batch = next(loader)
        key, _ = jax.random.split(key)

    n_obs = len(data_batch["map"])

    # The field posterior is huge: n_obs x (num_samples, 128, 128, 5) float32
    # ~= 65 GB for 100 obs. Stream it straight to a disk-backed .npy (memmap)
    # so we never hold more than one observation's samples (~0.65 GB) in RAM;
    # accumulating in a list + np.array() previously OOM-killed the job at save.
    x_mm = np.lib.format.open_memmap(
        f'{OUT_DIR}/x_samples_job_{job_id}.npy',
        mode='w+', dtype=np.float32,
        shape=(n_obs, num_samples, 128, 128, 5),
    )
    theta_prior = np.asarray(data_batch["theta"][:n_obs], dtype=np.float32)
    theta_posterior = np.empty((n_obs, num_samples, 6), dtype=np.float32)

    key, sk = jax.random.split(key)
    cond = make_cond(data_batch["map"], sk)

    for i in tqdm(range(n_obs)):
        key, sk = jax.random.split(key)
        x_samples, cosmo_samples = posterior_sampling(cond[i], sk)
        x_mm[i] = np.asarray(x_samples, dtype=np.float32)
        theta_posterior[i] = np.asarray(cosmo_samples, dtype=np.float32)

    x_mm.flush()
    del x_mm
    np.save(f'{OUT_DIR}/true_cosmo_job_{job_id}.npy', theta_prior)
    np.save(f'{OUT_DIR}/cosmo_samples_job_{job_id}.npy', theta_posterior)
    np.save(f'{OUT_DIR}/true_x_job_{job_id}.npy', np.asarray(data_batch["map"][:n_obs]))


if __name__ == "__main__":
    main()
