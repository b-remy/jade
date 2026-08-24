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

from datasets import load_from_disk

from functools import partial
from tqdm import tqdm

import yaml
import pickle

# LSST noise level
from jade.init import sigma_lsst

def main():

    # cfg, states = load_model(
    #     # "/u/bremy/repos/jade/experiments/wandb/run-20260218_132547-3685luj0/files/checkpoints",
    #     # Stage-2 split-QKV run, finetuned from e06v6sdj (cond_patch_size=8).
    #     "/u/bremy/repos/jade/experiments/wandb/run-20260504_100148-by4dv8sg/files/checkpoints",
    #     "JADE_B_16_ema_latest"
    #     #"JADE_B_16_latest"
    #     # "JADE_B_16_ema_best"
    # )

    # Match plot_amortized.py: same Stage-2 split-QKV checkpoint and 50-step Heun.
    cfg, states = load_model(
        #"/u/bremy/repos/jade/experiments/wandb/run-20260504_100148-by4dv8sg/files/checkpoints",
        "/u/bremy/repos/jade/experiments/wandb/run-20260507_170014-7hnur00g/files/checkpoints",
        "JADE_B_16_ema_best"
    )

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
    # train/test split used during training (see train_sharding.py:
    # train_test_split(test_size=val_split, seed=shuffle_seed)). Iterating the
    # full dataset here would draw ~95% in-sample (training) observations and
    # make the coverage look artificially good.
    dataset = load_from_disk("../sbi_lens_full")
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

    theta_prior = []
    theta_posterior = []
    x_posterior = []

    key, sk = jax.random.split(key)
    cond = make_cond(data_batch["map"], sk)

    for i in tqdm(range(len(data_batch["map"]))):
        key, sk = jax.random.split(key)
        x_samples, cosmo_samples = posterior_sampling(cond[i], sk)
        theta_prior.append(data_batch["theta"][i])
        theta_posterior.append(cosmo_samples)
        x_posterior.append(np.array(x_samples))

    theta_prior = np.array(theta_prior)
    theta_posterior = np.array(theta_posterior)
    x_posterior = np.array(x_posterior)
    np.save(f'tarp_results/conditional/true_cosmo_job_{job_id}.npy', theta_prior)
    np.save(f'tarp_results/conditional/cosmo_samples_job_{job_id}.npy', theta_posterior)
    np.save(f'tarp_results/conditional/true_x_job_{job_id}.npy', data_batch["map"])
    np.save(f'tarp_results/conditional/x_samples_job_{job_id}.npy', x_posterior)

    # # Save results for this job
    # with open(f'cosmo_samples_job_{job_id}.pkl', 'wb') as f:
    #     pickle.dump(results, f)

if __name__ == "__main__":
    main()

    
