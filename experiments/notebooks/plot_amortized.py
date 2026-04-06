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
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
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
    
    "/u/bremy/repos/jade/experiments/run-20260219_232046-jhj5rm2p/files/checkpoints",
    
    # "/u/bremy/repos/jade/experiments/wandb/run-20260221_001138-1g4mzv90/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260221_001420-y91pn6l8/files/checkpoints",
    
    
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

from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
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

x_samples_ = x_samples * FIELD_STD + FIELD_MEAN

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

# Theoretical Cl from jax_cosmo for the fiducial cosmology, evaluated at the
# lenstools bin centers. Used as the (noise-free) reference for the relative plot.
from sbi_lens.simulator.redshift import subdivide as _subdivide
_nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gals_per_arcmin2)
_nz_bins = _subdivide(_nz, nbins=nbins, zphot_sigma=0.05)
_tracer = jc.probes.WeakLensing(_nz_bins, sigma_e=sigma_e)
_cosmo_th = jc.Planck15(
    Omega_c=cosmo.Omega_c, Omega_b=cosmo.Omega_b, h=cosmo.h,
    n_s=cosmo.n_s, sigma8=cosmo.sigma8, w0=cosmo.w0,
)
_cl_th = np.array(jc.angular_cl.angular_cl(_cosmo_th, jnp.asarray(ell), [_tracer]))
# upper-triangular pair ordering: (0,0),(0,1),...,(0,4),(1,1),...,(4,4)
_pair_idx = {(min(i, j), max(i, j)): k
             for k, (i, j) in enumerate([(a_, b_) for a_ in range(5) for b_ in range(a_, 5)])}
ps_auto_th = np.stack([_cl_th[_pair_idx[(i, i)]] for i in range(5)])
ps_cross_th = np.zeros((5, 5, len(ell)))
for i in range(5):
    for j in range(5):
        if i != j:
            ps_cross_th[i, j] = _cl_th[_pair_idx[(min(i, j), max(i, j))]]

pix_size = 5 * 60 / 128 # arcmin / pixel
pix_size_rad =  np.pi * pix_size / (180 * 60)
print(pix_size_rad)
ell_max = 2*np.pi / (2*pix_size_rad)
print(ell_max)

ps_auto_samples = []
ps_cross_samples = []
n_ps_samples = min(64, len(x_samples_))
for s in tqdm(range(n_ps_samples), desc="computing ps"):
    ell_s, ps_auto_s, ps_cross_s = compute_ps(x_samples_[s], x_samples_[s])
    ps_auto_samples.append(ps_auto_s)
    ps_cross_samples.append(ps_cross_s)
ps_auto_samples = np.array(ps_auto_samples)
ps_cross_samples = np.array(ps_cross_samples)

ps_auto_mean = ps_auto_samples.mean(0)
ps_auto_std = ps_auto_samples.std(0)
ps_cross_mean = ps_cross_samples.mean(0)
ps_cross_std = ps_cross_samples.std(0)

from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

fontsize_text = 20  # For labels and legend
fontsize_ticks = 12  # For tick labels

fig, ax = plt.subplots(5,5,figsize=(10,10))
for i in range(5):
    for j in range(5):
        if j>i:
            ax[i, j].axis('off')
        else:
            if i==j:
                ax[i,j].loglog(ell, ps_auto[i], label='Fiducial', color="k", alpha=1.)
                ax[i,j].plot(ell, ps_auto_mean[i], color='tab:blue', alpha=1, label="Posterior mean")
                ax[i,j].fill_between(ell, ps_auto_mean[i] - ps_auto_std[i],
                                     ps_auto_mean[i] + ps_auto_std[i],
                                     color='tab:blue', alpha=0.3)
                ax[i,j].set_xscale('log')
                ax[i,j].set_yscale('log')
            else:
                ax[i,j].loglog(ell, ps_cross[:,i, j], color='k')
                ax[i,j].plot(ell, ps_cross_mean[:, i, j], color='tab:blue', alpha=1.)
                ax[i,j].fill_between(ell, ps_cross_mean[:, i, j] - ps_cross_std[:, i, j],
                                     ps_cross_mean[:, i, j] + ps_cross_std[:, i, j],
                                     color='tab:blue', alpha=0.3)
                ax[i,j].set_xscale('log')
                ax[i,j].set_yscale('log')
            ax[i,j].set_xlim(ell.min(), ell.max())
            ax[i,j].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
            ax[i,j].yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax[i,j].xaxis.set_minor_formatter(NullFormatter())
            ax[i,j].yaxis.set_minor_formatter(NullFormatter())
        
        if i==4:
            ax[i,j].tick_params(axis='x', labelsize=fontsize_ticks)
        else:
            ax[i,j].tick_params(axis='x', labelbottom=False)

        if j==0:
            ax[i,j].tick_params(axis='y', labelsize=fontsize_ticks)
        else:
            ax[i,j].tick_params(axis='y', labelleft=False)

fig.supxlabel(r'$\ell$', fontsize=fontsize_text)
fig.supylabel(r'$\mathcal{C}_\ell$', fontsize=fontsize_text)

# At the end, after the loops
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.88, 0.88), fontsize=18)
plt.savefig(os.path.join(save_dir,"power-spectra-posterior.png"), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(save_dir,"power-spectra-posterior.pdf"), bbox_inches='tight', pad_inches=0)
plt.close(fig)


# Relative power spectrum: (Cl_th - Cl_sample) / Cl_th, using the theoretical
# (jax_cosmo) Cl as reference instead of the noisy single realization.
rel_auto_samples = (ps_auto_th[None] - ps_auto_samples) / ps_auto_th[None]
# ps_cross_samples has shape (n_samples, n_ell, 5, 5); ps_cross_th is (5, 5, n_ell)
ps_cross_th_for_rel = np.transpose(ps_cross_th, (2, 0, 1))[None]  # (1, n_ell, 5, 5)
# avoid divide-by-zero on the diagonal of ps_cross_th (auto entries are 0 there)
_safe = np.where(ps_cross_th_for_rel == 0, 1.0, ps_cross_th_for_rel)
rel_cross_samples = (ps_cross_th_for_rel - ps_cross_samples) / _safe

rel_auto_mean = rel_auto_samples.mean(0)
rel_auto_std = rel_auto_samples.std(0)
rel_cross_mean = rel_cross_samples.mean(0)
rel_cross_std = rel_cross_samples.std(0)

fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        if j > i:
            ax[i, j].axis('off')
        else:
            if i == j:
                ax[i, j].axhline(0, color="k", alpha=1., label='Fiducial')
                ax[i, j].plot(ell, rel_auto_mean[i], color='tab:blue', alpha=1, label="Posterior mean")
                ax[i, j].fill_between(ell, rel_auto_mean[i] - rel_auto_std[i],
                                      rel_auto_mean[i] + rel_auto_std[i],
                                      color='tab:blue', alpha=0.3)
                ax[i, j].set_xscale('log')
            else:
                ax[i, j].axhline(0, color='k')
                ax[i, j].plot(ell, rel_cross_mean[:, i, j], color='tab:blue', alpha=1.)
                ax[i, j].fill_between(ell, rel_cross_mean[:, i, j] - rel_cross_std[:, i, j],
                                      rel_cross_mean[:, i, j] + rel_cross_std[:, i, j],
                                      color='tab:blue', alpha=0.3)
                ax[i, j].set_xscale('log')
            ax[i, j].set_xlim(ell.min(), ell.max())
            ax[i, j].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
            ax[i, j].yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax[i, j].xaxis.set_minor_formatter(NullFormatter())

        if i == 4:
            ax[i, j].tick_params(axis='x', labelsize=fontsize_ticks)
        else:
            ax[i, j].tick_params(axis='x', labelbottom=False)

        if j == 0:
            ax[i, j].tick_params(axis='y', labelsize=fontsize_ticks)
        else:
            ax[i, j].tick_params(axis='y', labelleft=False)

fig.supxlabel(r'$\ell$', fontsize=fontsize_text)
fig.supylabel(r'$(\mathcal{C}_\ell^{\rm truth}-\mathcal{C}_\ell^{\rm sample})/\mathcal{C}_\ell^{\rm truth}$', fontsize=fontsize_text)

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.88, 0.88), fontsize=18)
plt.savefig(os.path.join(save_dir, "power-spectra-posterior-relative.png"), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(save_dir, "power-spectra-posterior-relative.pdf"), bbox_inches='tight', pad_inches=0)
plt.close(fig)

# Average relative error per auto bin and overall
# Use mean across posterior samples, then absolute value, then average over ell
avg_rel_err_per_bin = np.mean(np.abs(rel_auto_mean), axis=1)  # shape (5,)
avg_rel_err_overall = np.mean(np.abs(rel_auto_mean))

print("Average relative error per auto bin:")
for i, err in enumerate(avg_rel_err_per_bin):
    print(f"  bin {i}: {err:.4f}")
print(f"Overall average relative error (auto): {avg_rel_err_overall:.4f}")


# One-point function of the convergence fields per z-bin.
# Use KDEs (smooth, paper-friendly) instead of histograms. Plot the truth as a
# black line and posterior samples as a blue mean ± 1 sigma band.
from scipy.stats import gaussian_kde

n_kde_samples = min(64, len(x_samples_))

fig, ax = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
for i in range(5):
    truth_vals = np.asarray(m_data[..., i]).ravel()
    lo, hi = np.percentile(truth_vals, [0.1, 99.9])
    pad = 0.1 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)

    kde_truth = gaussian_kde(truth_vals)
    pdf_truth = kde_truth(grid)

    pdf_samples = np.zeros((n_kde_samples, grid.size))
    for s in range(n_kde_samples):
        vals = np.asarray(x_samples_[s, :, :, i]).ravel()
        pdf_samples[s] = gaussian_kde(vals)(grid)
    pdf_mean = pdf_samples.mean(0)
    pdf_std = pdf_samples.std(0)

    ax[i].plot(grid, pdf_truth, color='k', label='Fiducial')
    ax[i].plot(grid, pdf_mean, color='tab:blue', label='Posterior mean')
    ax[i].fill_between(grid, pdf_mean - pdf_std, pdf_mean + pdf_std,
                       color='tab:blue', alpha=0.3)
    ax[i].set_title(f'bin {i + 1}', fontsize=fontsize_text)
    ax[i].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[i].xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[i].set_xlim(grid[0], grid[-1])

fig.supxlabel(r'$\kappa$', fontsize=fontsize_text)
fig.supylabel(r'$p(\kappa)$', fontsize=fontsize_text)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=len(labels), fontsize=18, frameon=False)
plt.savefig(os.path.join(save_dir, "one-point-function.png"), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(save_dir, "one-point-function.pdf"), bbox_inches='tight', pad_inches=0)
plt.close(fig)

