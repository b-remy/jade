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

from jade.nn import JADE_B_16, JADE_M_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.flow import Denoiser, PosteriorDenoiser
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
sigma_e = 0.26
n_gal = 27 # galaxies per arcmin^2
A_pix = 5.49 # arcsin^2
N_s = n_gal * A_pix

sigma_noise = sigma_e / math.sqrt(N_s)

def main():

    cfg, states = load_model(
        #"/u/bremy/repos/jade/experiments/wandb/run-20251225_123135-24tt5plt/files/checkpoints", 
        #"/u/bremy/repos/jade/experiments/wandb/run-20251227_054406-n87abk5f/files/checkpoints",
        #"/u/bremy/repos/jade/experiments/wandb/run-20260104_084145-r9ebud0a/files/checkpoints",
        "/u/bremy/repos/jade/experiments/wandb/run-20260104_113137-xqaq0qgl/files/checkpoints",
        #"/u/bremy/repos/jade/experiments/wandb/latest-run/files/checkpoints",
        
        #"JADE_B_16_ema_latest"
        "JADE_B_16_ema_best"
    )

    model = JADE_B_16(
        rngs=nnx.Rngs(cfg['training']['seed']),
        in_channels=cfg['model']['in_channels'],
        input_size=cfg['model']['input_size'],
        patch_size=16)

    model = Denoiser(model, cfg)
    #model = FlowDenoiser(model, cfg)
    nnx.update(model, states)

    @partial(jax.jit, static_argnames=['batch_size'])
    def posterior_sampling(batch, i, key, batch_size=200):
        kappa = batch["map"][i]

        # compute gamma from ith kappa map in the batch
        gamma1, gamma2 = ks93inv(kappa, jnp.zeros_like(kappa))
        gamma = jnp.stack([gamma1, gamma2])

        # add noise
        key, subkey = jax.random.split(key)
        noise = sigma_noise * jax.random.normal(shape=(2,128,128,5), key=subkey)
        obs = gamma + noise

        pdenoiser = PosteriorDenoiser(model=model, cfg=cfg, gamma=gamma, sigma_gamma=sigma_noise)
        sampler = EulerSampler(pdenoiser, 512)

        keys = jax.random.split(key, 3)
        x_0 = jax.random.normal(keys[0], shape=(batch_size, 128, 128, 5))
        cosmo_0 = jax.random.normal(keys[1], shape=(batch_size, 6))

        keys = jax.random.split(keys[2], batch_size)
        x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, keys)
        x_samples = x_samples * FIELD_STD + FIELD_MEAN
        cosmo_samples = cosmo_samples * THETA_STD + THETA_MEAN

        return x_samples, cosmo_samples
        
    dataset = load_from_disk("sbi_lens_lognormal")
    dataset = dataset.with_format("numpy")
    
    job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    n_jobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 20))

    print(f"Starting job {job_id} out of {n_jobs}")

    # Total number of iterations n_sims
    # total_iterations = 500
    total_iterations = 400

    # Calculate which iterations this job handles
    # number of prior samples in the dataset per job
    iters_per_job = total_iterations // n_jobs

    # batch_size = 100 # number of prior samples in the dataset per job
    loader = dataset.iter(batch_size=iters_per_job)

    start_idx = job_id * iters_per_job

    # Last job handles any remainder
    if job_id == n_jobs - 1:
        end_idx = total_iterations
    else:
        end_idx = start_idx + iters_per_job

    print(f"Job {job_id}: Processing iterations {start_idx} to {end_idx-1}")

    data_batch = next(loader)
    for _ in range(job_id):    
        data_batch = next(loader)        

     # Create base random key - use a fixed seed for reproducibility
    base_seed = 42
    master_key = jax.random.key(base_seed)
    
    # Split keys for all iterations upfront
    # This ensures keys are identical regardless of how work is split
    
    job_keys = jax.random.split(master_key, n_jobs)
    job_key = job_keys[job_id]

    key, _ = jax.random.split(job_key)

    # Process only this job's subset of iterations
    results_cosmo = []
    results_x = []
    # for i in tqdm(range(start_idx, end_idx)):
    for i in tqdm(range(iters_per_job)):
        post_cosmo = []
        post_x = []
        for n_batches in range(2):
            key, subkey = jax.random.split(key)
            x_samples, cosmo_samples = posterior_sampling(data_batch, i, subkey, batch_size=200)
            post_x.append(x_samples)
            post_cosmo.append(cosmo_samples)

        post_cosmo = jnp.concatenate(post_cosmo, axis=0)
        post_x = jnp.concatenate(post_x, axis=0)

        results_cosmo.append(np.array(post_cosmo))
        results_x.append(np.array(post_x))

    results_cosmo = np.array(results_cosmo)
    results_x = np.array(results_x)

    np.save(f'tarp_results/512/true_cosmo_job_{job_id}.npy', data_batch["theta"])
    np.save(f'tarp_results/512/true_x_job_{job_id}.npy', data_batch["map"])
    np.save(f'tarp_results/512/cosmo_samples_job_{job_id}.npy', results_cosmo)
    np.save(f'tarp_results/512/x_samples_job_{job_id}.npy', results_x)

    # # Save results for this job
    # with open(f'cosmo_samples_job_{job_id}.pkl', 'wb') as f:
    #     pickle.dump(results, f)

if __name__ == "__main__":
    main()

    