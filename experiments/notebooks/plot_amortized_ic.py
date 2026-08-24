import os
import argparse
import yaml

import jax
import jax.numpy as jnp

from flax import nnx
import orbax.checkpoint as ocp

import matplotlib.pyplot as plt
import numpy as np

# from jade.nn_conditional import JADE_B_16
from jade.nn_hybrid import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, GRF_MEAN, GRF_STD
from jade.flow import Denoiser
from jade.utils import load_model
from train import plot_denoiser, normalize_batch

from datasets import load_from_disk

import yaml
import pickle

from lenstools import ConvergenceMap

print(jax.devices())

from jade.init import sigma_lsst
print("sigma_lsst", sigma_lsst)

from jade.jade_fuse import JADE_FUSE_B_16_mixed
from jade.nn_patch import JADE_B_16_mixed

save_dir = 'amortized'

cfg, states = load_model(
    # "/u/bremy/repos/jade/experiments/wandb/run-20260210_174412-mzczznxv/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260211_134514-3lgcf0yj/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260211_231954-bzn2meri/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260212_161618-mvdbkagz/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260213_094725-gvlkj3u8/files/checkpoints",
    
    # "/u/bremy/repos/jade/experiments/wandb/run-20260216_003019-3afzmbtk/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260216_113125-oo0qj7m5/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260216_215834-uvlfixxc/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260217_144342-cd9l9a8s/files/checkpoints",

    # "/u/bremy/repos/jade/experiments/wandb/run-20260217_173825-gbm2flrd/files/checkpoints",
    #  "/u/bremy/repos/jade/experiments/wandb/run-20260218_132547-3685luj0/files/checkpoints",
    
    # "/u/bremy/repos/jade/experiments/wandb/run-20260219_232046-jhj5rm2p/files/checkpoints",
    
    # "/u/bremy/repos/jade/experiments/wandb/run-20260221_001138-1g4mzv90/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260221_001420-y91pn6l8/files/checkpoints",
    
    # "/u/bremy/repos/jade/experiments/wandb/run-20260226_093505-3gytbpbd/files/checkpoints",
    "/u/bremy/repos/jade/experiments/wandb/run-20260302_104833-pqwyma03/files/checkpoints",
    
    "JADE_B_16_ema_latest"
    #"JADE_B_16_latest"
    # "JADE_B_16_ema_best"
)

SCALE_COSMO = 1

# model = JADE_FUSE_B_16_mixed(
#         rngs=nnx.Rngs(cfg['training']['seed']), 
#         in_channels=cfg['model']['in_channels'], 
#         input_size=cfg['model']['input_size'],
#         enable_cond_image=cfg["model"]["enable_cond_image"],
#         cond_channels=cfg["model"]["cond_channels"],
#         # patch_size=cfg["model"]["patch_size"]
#     )

# model = JADE_B_16_mixed(
#         rngs=nnx.Rngs(cfg['training']['seed']), 
#         in_channels=cfg['model']['in_channels'], 
#         input_size=cfg['model']['input_size'],
#         enable_cond_image=cfg["model"]["enable_cond_image"],
#         cond_channels=cfg["model"]["cond_channels"],
#         num_cosmo_tokens=cfg['model']['num_cosmo_tokens'],
#         cond_patch_size=cfg['model']['cond_patch_size'],
#         # patch_size=cfg["model"]["patch_size"]
#     )


# model = JADE_B_16(
#         rngs=nnx.Rngs(cfg['training']['seed']), 
#         in_channels=cfg['model']['in_channels'], 
#         input_size=cfg['model']['input_size'],
#         enable_cond_image=cfg["model"]["enable_cond_image"],
#         cond_channels=cfg["model"]["cond_channels"],
#         num_cosmo_tokens=cfg['model']['num_cosmo_tokens'],
#         cond_patch_size=cfg['model']['cond_patch_size'],
#         cond_start=cfg['model']['cond_start'],
#         patch_size=cfg["model"]["patch_size"]
#     )

model = JADE_B_16(
        rngs=nnx.Rngs(cfg['training']['seed']), 
        in_channels=cfg['model']['in_channels'], 
        input_size=cfg['model']['input_size'],
        enable_cond_image=cfg["model"]["enable_cond_image"],
        cond_channels=cfg["model"]["cond_channels"],
        num_cosmo_tokens=cfg['model']['num_cosmo_tokens'],
        cond_patch_size=cfg['model']['cond_patch_size'],
        cond_start=cfg['model']['cond_start'],
        attn_drop=cfg['model']['attn_drop'],
        proj_drop=cfg['model']['proj_drop'],
    )

print("cfg['loss']['SCALE_COSMO']", cfg['loss']["SCALE_COSMO"])

model = Denoiser(model, cfg)
#model = FlowDenoiser(model, cfg)
nnx.update(model, states)

import pickle
import os

mcmc_dir = "./mcmc_log_normal_sample/good_noise_level"

# Load the dictionary
with open(os.path.join(mcmc_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
    data = pickle.load(f)

with open(os.path.join(mcmc_dir, "mcmc_log_posterior_samples.pkl"), "rb") as f:
    samples = pickle.load(f)

plt.figure()
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(data['y'][...,i])
    plt.colorbar()
plt.savefig(os.path.join(save_dir, "observation.png"))

print("obs saved")

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

from jade.sampling import EulerSampler, HeunSampler

# @jax.jit
def sample(obs, key, states=states, batch_size=128):
    nnx.update(model, states)
    sampler = EulerSampler(model=model, num_steps=128)
    # sampler = HeunSampler(model=model, num_steps=256)
    keys = jax.random.split(key, 3)
    x_0 = jax.random.normal(keys[0], shape=(batch_size, 128, 128, 5))
    cosmo_0 = jax.random.normal(keys[1], shape=(batch_size, 6))
    
    keys = jax.random.split(keys[2], batch_size)
    
    cond = (obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)
    #cond = obs / sigma_lsst.reshape((1,1,-1))
    #cond = (data['y'] - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)

    x_samples, cosmo_samples = jax.vmap(sampler, in_axes=(0,0,None,0))(x_0, cosmo_0, cond, keys)

    return x_samples, cosmo_samples

from tqdm import tqdm

obs = data['y']

key = jax.random.key(0)

x_samples, cosmo_samples = sample(obs, key, states)

for i in tqdm(range(3)):
    key, subkey = jax.random.split(key)
    x_samples_, cosmo_samples_ = sample(obs, subkey, states)
    x_samples = jnp.concatenate([x_samples, x_samples_])
    cosmo_samples = jnp.concatenate([cosmo_samples, cosmo_samples_])

from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, GRF_MEAN, GRF_STD
from getdist import MCSamples, plots

names = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]
samples_mcmc = MCSamples(samples=np.array(samples), names=names, label="mcmc")

theta_posterior = (cosmo_samples / SCALE_COSMO * THETA_STD + THETA_MEAN )
samples_posterior = MCSamples(samples=np.array(theta_posterior), names=names, label="Diffusion")

# Triangle plot (sometimes also called a corner plot)
# g = plots.get_subplot_plotter()
# g.triangle_plot([samples_posterior, samples_mcmc], names, markers=data['theta'], marker_args={"lw": 1}, filled=True)

# plt.savefig(os.path.join(save_dir, "contour_plot.png"))
# Triangle plot (sometimes also called a corner plot)
g = plots.get_subplot_plotter()

# Increase font sizes
g.settings.axes_fontsize = 26       # Tick labels
g.settings.axes_labelsize = 28      # Axis labels
g.settings.legend_fontsize = 28     # Legend text

# samples_mcmc = MCSamples(samples=np.array(samples), names=names, label="MCMC")
# samples_posterior = MCSamples(samples=np.array(theta_posterior), names=names, label="Diffusion")

g.triangle_plot(
    [samples_posterior, samples_mcmc], 
    names, 
    markers=data['theta'], 
    marker_args={"lw": 1}, 
    filled=[True, False],
    line_args=[
        {"ls": "-", "color": "#d06e99ff"},  # Diffusion: solid line with your color
        {"ls": "--", "color": "black"}       # MCMC: dashed black line
    ],
    alpha=[1.,0.1],
    contour_colors=["#d06e99ff", "black"],  # Colors for 2D contours
    contour_ls=["-", "--"],  # Line styles for 2D contours
    contour_lws=[2., 2.]  # Line widths for 2D contours
)
plt.savefig(os.path.join(save_dir, "contour_plot.png"))
plt.savefig(os.path.join(save_dir, "contour_plot.pdf"))

print("contours saved")

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
with_noise = False

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

key = jax.random.key(0)
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
data_no_noise = {
    "theta": jnp.stack(
        [model_trace[name]["value"] for name in params_name], axis=-1
    ),
    "y": model_trace["y"]["value"],
}

import astropy.units as u
from lenstools import ConvergenceMap

l_edges_kmap= np.linspace(500, 4608.0, 128)

import itertools
def fill_lower_diag(array,nl):
    n = int(np.sqrt(len(array)*2))+1
    mask = np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n, nl))
    out[np.stack(mask,axis=1)] = array
    return out.T

x_samples_ = x_samples * GRF_STD + GRF_MEAN

m_data = data_no_noise["y"]
map_size = 5

def compute_ps(m_data1, m_data_2):
    lis=[0,1,2,3,4]
    p_cross = []
    
    for i, j in itertools.combinations(lis, 2):
        ell, ps = ConvergenceMap(
            m_data1[:,:,i], 
            angle=map_size*u.deg
        ).cross(
            ConvergenceMap(
                m_data_2[:,:,j], 
               angle=map_size*u.deg),
            l_edges=l_edges_kmap)
        p_cross.append(ps)
        
    ps_cross=np.array(p_cross)
    ps_cross = fill_lower_diag(ps_cross, 127)
    
    ps_auto=[]
    for i in range(5):
        ell, pi = ConvergenceMap(
            m_data1[:,:,i], 
            angle=map_size*u.deg
        ).cross(ConvergenceMap(m_data_2[:,:,i], angle=map_size*u.deg),l_edges=l_edges_kmap)
        ps_auto.append(pi)
    ps_auto = np.array(ps_auto)
    return ell, ps_auto, ps_cross

ell, ps_auto, ps_cross = compute_ps(m_data, m_data)

pix_size = 5 * 60 / 128 # arcmin / pixel
pix_size_rad =  np.pi * pix_size / (180 * 60)
print(pix_size_rad)
ell_max = 2*np.pi / (2*pix_size_rad)
print(ell_max)

index = np.argmin(((cosmo_samples-data["theta"])**2).sum(1))

ps_auto_samples = []
ps_cross_samples = []
ell_s, ps_auto_s, ps_cross_s = compute_ps(x_samples_[index], x_samples_[index])
ps_auto_samples.append(ps_auto_s)
ps_cross_samples.append(ps_cross_s)
ps_auto_samples = np.array(ps_auto_samples)
ps_cross_samples = np.array(ps_cross_samples)

fontsize_text = 20  # For labels and legend
fontsize_ticks = 16  # For tick labels

fig, ax = plt.subplots(5,5,figsize=(10,10))
for i in range(5):
    for j in range(5):
        if j>i:
            ax[i, j].axis('off')
        else:
            if i==j:
                ax[i,j].loglog(ell, ps_auto[i], label='Fiducial', color="k", alpha=1.)
                for ps in ps_auto_samples:
                    ax[i,j].loglog(ell, ps[i], color='tab:blue', alpha=1, label="Posterior sample")
                
                
            else:
                ax[i,j].loglog(ell, ps_cross[:,i, j], color='k')
                for ps in ps_cross_samples:
                    ax[i,j].loglog(ell, ps[:,i,j], color='tab:blue', alpha=1.)
        
        # Show x-axis label and ticks only on bottom row
        if i==4:
            ax[i,j].set_xlabel(r'$\ell$', fontsize=fontsize_text)
            ax[i,j].tick_params(axis='x', labelsize=fontsize_ticks)
        else:
            ax[i,j].tick_params(axis='x', labelbottom=False)
        
        # Show y-axis label and ticks only on left column
        if j==0:
            ax[i,j].set_ylabel(r'$\mathcal{C}_\ell$', fontsize=fontsize_text)
            ax[i,j].tick_params(axis='y', labelsize=fontsize_ticks)
        else:
            ax[i,j].tick_params(axis='y', labelleft=False)

# At the end, after the loops
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.88, 0.88), fontsize=18)
plt.savefig(os.path.join(save_dir,"power-spectra-posterior.png"), bbox_inches='tight', pad_inches=0)
                                        
