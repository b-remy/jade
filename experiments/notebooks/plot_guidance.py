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
from jade.lensing import ks93inv

save_dir = 'guidance'

cfg, states = load_model(
    # "/u/bremy/repos/jade/experiments/wandb/run-20260210_001031-lv23f4ai/files/checkpoints",
    #"/u/bremy/repos/jade/experiments/wandb/run-20260210_142535-w00jwh44/files/checkpoints",
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

SCALE_COSMO = cfg.get('loss', {}).get('SCALE_COSMO', 1.0)
print("SCALE_COSMO", SCALE_COSMO)

model = Denoiser(model, cfg)
#model = FlowDenoiser(model, cfg)
nnx.update(model, states)

from jade.flow import PosteriorDenoiser

# dataset = load_from_disk("sbi_lens_lognormal")
# dataset = dataset.with_format("numpy")
# dataset = dataset.train_test_split(
#         test_size=cfg['data']['val_split'], 
#         seed=cfg['data']['shuffle_seed']
#     )["test"]
# batch_size = 64
# loader = dataset.iter(batch_size=batch_size)

import pickle
import os

mcmc_dir = "./mcmc_log_normal_sample/good_noise_level"

# Load the dictionary
#with open(os.path.join(save_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
#    data = pickle.load(f)

with open(os.path.join(mcmc_dir, "mcmc_log_posterior_samples.pkl"), "rb") as f:
    samples_mcmc = pickle.load(f)

import sbi_lens
import numpyro
import numpyro.distributions as dist
import jax_cosmo as jc

cosmo = jc.parameters.Planck15()

key = jax.random.key(0)

from numpyro import sample
from numpyro.handlers import condition, reparam, seed, trace
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from functools import partial

from numpyro import sample
from numpyro.handlers import condition, reparam, seed, trace
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from functools import partial

sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b = config_lsst_y_10.b
z0 = config_lsst_y_10.z0
N = 128
map_size = 5
with_noise = True

model_log_normal = partial(
    lensingLogNormal,
    N=N,
    map_size=map_size,
    gal_per_arcmin2=gals_per_arcmin2,
    sigma_e=sigma_e,
    nbins=nbins,
    a=a,
    b=b,
    z0=z0,
    model_type='lognormal',
    lognormal_shifts='LSSTY10',
    with_noise=with_noise,
    )

cond_model = seed(model_log_normal, key)
cond_model = condition(
    cond_model,
    {
        "omega_c": cosmo.Omega_c,
        "omega_b": cosmo.Omega_b,
        "sigma_8": cosmo.sigma8,
        "h_0": cosmo.h,
        "n_s": cosmo.n_s,
        "w_0": cosmo.w0,
    },
)

params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]

model_trace = trace(cond_model).get_trace()
data = {
    "theta": jnp.stack(
        [model_trace[name]["value"] for name in params_name], axis=-1
    ),
    "y": model_trace["y"]["value"],
}

obs = data['y']

from jade.init import sigma_lsst
print("sigma_lsst", sigma_lsst)

gamma1, gamma2 = ks93inv(obs, jnp.zeros_like(obs))
gamma = jnp.stack([gamma1, gamma2])

pdenoiser = PosteriorDenoiser(model=model, cfg=cfg, gamma=gamma, sigma_gamma=sigma_lsst)

from jade.sampling import EulerSampler, HeunSampler

sampler = EulerSampler(pdenoiser, 512)
# sampler = HeunSampler(pdenoiser, 512)

from functools import partial

#@partial(jax.jit, static_argnums=(1,))
def get_posterior_sampels(key, batch_size=128):
    keys = jax.random.split(key, 3)

    x_0 = jax.random.normal(keys[0], shape=(batch_size, 128, 128, 5))
    cosmo_0 = jax.random.normal(keys[1], shape=(batch_size, 6))

    keys = jax.random.split(keys[3], batch_size)
    x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, None, keys)
    return x_samples, cosmo_samples

key = jax.random.key(0)
kappa_, theta_ = get_posterior_sampels(key)

from tqdm import tqdm

# for i in tqdm(range(2)):
#     key, subkey = jax.random.split(key)
#     kappa_b, theta_b = get_posterior_sampels(subkey, 60)
#     theta_ = jnp.concatenate([theta_, theta_b], 0)

from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD

from getdist import MCSamples, plots
names = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]

theta_posterior = theta_ / SCALE_COSMO * THETA_STD + THETA_MEAN
samples_posterior = MCSamples(samples=np.array(theta_posterior), names=names, label="Diffusion")

names = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]
samples_mcmc = MCSamples(samples=np.array(samples_mcmc), names=names, label="mcmc")

# Triangle plot (sometimes also called a corner plot)
g = plots.get_subplot_plotter()
#g.settings.smooth_scale_2D = 0.5  # adjust between 0.3-1.0
g.triangle_plot([samples_posterior, samples_mcmc], names, markers=data['theta'], marker_args={"lw": 1}, filled=True)

plt.savefig(os.path.join(save_dir, "contour_plot.png"))
