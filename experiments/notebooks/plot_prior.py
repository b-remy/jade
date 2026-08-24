import os
import argparse
import yaml

import jax
import jax.numpy as jnp

from flax import nnx
import orbax.checkpoint as ocp

import matplotlib.pyplot as plt
import numpy as np

from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.nn_one_token import JADE_B_16 
from jade.flow import Denoiser
#from jade.nn_conditional import JADE_B_16 
#from jade.flow_conditional import Denoiser


from jade.utils import load_model
from train import plot_denoiser, normalize_batch


from datasets import load_from_disk

import yaml
import pickle

from jade.nn_patch import JADE_B_16_mixed

from tqdm import tqdm

save_dir = 'prior'

cfg, states = load_model(
    # "/u/bremy/repos/jade/experiments/wandb/run-20260210_001031-lv23f4ai/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260210_142535-w00jwh44/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260211_160225-7ka2hhmx/files/checkpoints",
    "/u/bremy/repos/jade/experiments/wandb/run-20260212_212841-9o3e214s/files/checkpoints",
    "JADE_B_16_ema_latest"
    #"JADE_B_16_latest"
    #"JADE_B_16_ema_best"
)

SCALE_COSMO = cfg['loss']['SCALE_COSMO']
print("SCALE_COSMO", SCALE_COSMO)

model = JADE_B_16_mixed(
        rngs=nnx.Rngs(cfg['training']['seed']), 
        in_channels=cfg['model']['in_channels'], 
        input_size=cfg['model']['input_size'],
        enable_cond_image=cfg["model"]["enable_cond_image"],
        cond_channels=cfg["model"]["cond_channels"],
        # patch_size=cfg["model"]["patch_size"]
    )

model = Denoiser(model, cfg)
#model = FlowDenoiser(model, cfg)

nnx.update(model, states)

from jade.sampling import EulerSampler, HeunSampler

sampler = EulerSampler(model, 256)

key = jax.random.key(0)
key, subkey = jax.random.split(key)

batch_size = 128

@jax.jit
def sample(key):
    
    """
    x_samples, cosmo_samples = model.generate(key, 
                batch_size=512, 
                x_shape=x_val, 
                cosmo_shape=cosmo_val, 
                use_ve=False
                )
    """
    keys = jax.random.split(key, 3)

    x_0 = jax.random.normal(keys[0], shape=(batch_size, 128, 128, 5))
    cosmo_0 = jax.random.normal(keys[1], shape=(batch_size, 6))

    keys = jax.random.split(keys[3], batch_size)
    #x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, keys)
    x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, None, keys)
    return x_samples, cosmo_samples
    
    #return x_samples, cosmo_samples

x_samples, cosmo_samples = sample(subkey)

for k in tqdm(range(5)):
    key, subkey = jax.random.split(key)
    _, cosmo_samples_ = sample(subkey)
    cosmo_samples = np.concatenate([cosmo_samples, cosmo_samples_], 0)

from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD

for i in range(5):
    plt.figure(figsize=(15,4))
    for j in range(5):
        plt.subplot(1,5,j+1)
        plt.imshow((x_samples[i,...,j]))
        plt.colorbar()
    plt.show()
    plt.savefig(os.path.join(save_dir, f"field_{i}.png"))

# cosmo = cosmo_samples / SCALE_COSMO * THETA_STD + THETA_MEAN 

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random

def cosmology_prior_model():
    """Define the joint prior distribution for cosmological parameters."""
    omega_c = numpyro.sample("omega_c", dist.TruncatedNormal(0.2664, 0.2, low=0))
    omega_b = numpyro.sample("omega_b", dist.Normal(0.0492, 0.006))
    sigma_8 = numpyro.sample("sigma_8", dist.Normal(0.831, 0.14))
    h_0 = numpyro.sample("h_0", dist.Normal(0.6727, 0.063))
    n_s = numpyro.sample("n_s", dist.Normal(0.9645, 0.08))
    w_0 = numpyro.sample("w_0", dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3))
    
    return {
        "omega_c": omega_c,
        "omega_b": omega_b,
        "sigma_8": sigma_8,
        "h_0": h_0,
        "n_s": n_s,
        "w_0": w_0
    }

def sample_priors_batch(rng_key, num_samples):
    """
    Sample from the joint prior distribution in batches.
    
    Args:
        rng_key: JAX random key
        num_samples: Number of samples to draw
    
    Returns:
        Dictionary with parameter names as keys and arrays of samples as values
    """
    # Use numpyro.infer.Predictive to sample from the prior
    from numpyro.infer import Predictive
    
    predictive = Predictive(cosmology_prior_model, num_samples=num_samples)
    samples = predictive(rng_key)
    
    return samples

rng_key = jax.random.key(0)

# Sample 1000 parameter sets
batch_size = 10_000
prior_samples = sample_priors_batch(rng_key, batch_size)

names = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]

cosmo_samples_vec = cosmo_samples / SCALE_COSMO * THETA_STD + THETA_MEAN

plt.figure(figsize=(10,6))
for k in range(6):
    plt.subplot(2,3,k+1)
    plt.hist(cosmo_samples_vec[:,k], bins=32, density=True, alpha=0.5, label="Diffusion samples");
    plt.title(names[k])
    
    if names[k]=="$\\Omega_c$":
        plt.hist(prior_samples["omega_c"], bins=32, density=True, alpha=0.5, label="Prior samples")
        
    if names[k]=="$\\Omega_b$":
        plt.hist(prior_samples["omega_b"], bins=32, density=True, alpha=0.5)
    if names[k]=="$\\sigma_8$":
        plt.hist(prior_samples["sigma_8"], bins=32, density=True, alpha=0.5)
    if names[k]=="$h_0$":
        plt.hist(prior_samples["h_0"], bins=32, density=True, alpha=0.5)
    if names[k]=="$n_s$":
        plt.hist(prior_samples["n_s"], bins=32, density=True, alpha=0.5)
    if names[k]=="$w_0$":
        plt.hist(prior_samples["w_0"], bins=32, density=True, alpha=0.5)
    
    if k==0:
        plt.legend()
        
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "histogram.png"))

prior_samples_ = np.stack([
    prior_samples["omega_c"],
    prior_samples["omega_b"],
    prior_samples["sigma_8"],
    prior_samples["h_0"],
    prior_samples["n_s"],
    prior_samples["w_0"]
    ], 1) 

from getdist import MCSamples, plots

samples = MCSamples(samples=np.array(cosmo_samples_vec), label="diffusion")

prior_samples = MCSamples(samples=prior_samples_, label="prior")

# Triangle plot (sometimes also called a corner plot)
g = plots.get_subplot_plotter()
#g.settings.smooth_scale_2D = 0.5  # adjust between 0.3-1.0
g.triangle_plot([samples, prior_samples], filled=True)
plt.savefig(os.path.join(save_dir, "contour_plot.png"))