import os
import argparse
import yaml

import jax
import jax.numpy as jnp

from flax import nnx
import orbax.checkpoint as ocp

import matplotlib as mpl
import matplotlib.pyplot as plt
# Paper-style typography without a system TeX install. ``mathtext.fontset='cm'``
# uses Computer Modern (bundled with matplotlib) for everything inside $...$,
# while plain text falls back to whatever serif font is available (DejaVu Serif
# on most clusters). All math labels in this script are already wrapped in
# $...$, so the figures end up looking close to pdflatex output.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})
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
    
    
    # "/u/bremy/repos/jade/experiments/wandb/run-20260221_001138-1g4mzv90/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260221_001420-y91pn6l8/files/checkpoints",
    
    
    # "/u/bremy/repos/jade/experiments/run-20260219_232046-jhj5rm2p/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260414_223438-2kx88ezd/files/checkpoints",
    # "/u/bremy/repos/jade/experiments/wandb/run-20260414_232810-e06v6sdj/files/checkpoints",
    # Stage-2 split-QKV run, finetuned from e06v6sdj (cond_patch_size=8).
    # "/u/bremy/repos/jade/experiments/wandb/run-20260504_100148-by4dv8sg/files/checkpoints",

     "/u/bremy/repos/jade/experiments/wandb/run-20260507_170014-7hnur00g/files/checkpoints",
    #"/u/bremy/repos/jade/experiments/wandb/run-20260511_095814-kj94osc8/files/checkpoints",

    #"JADE_B_16_ema_latest"
     "JADE_B_16_latest"
    #"JADE_B_16_ema_best"
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
        # Stage-2 runs set this; older stage-1 configs don't have the key.
        split_qkv=cfg['model'].get('split_qkv', False),
        mask_theta_to_field=cfg['model'].get('mask_theta_to_field', False),
    )

print("cfg['loss']['SCALE_COSMO']", cfg['loss']["SCALE_COSMO"])

model = Denoiser(model, cfg)
#model = FlowDenoiser(model, cfg)
nnx.update(model, states)

# Sampler-side t_eps override: with the mixture-finetuned model, the score
# field is well-trained at t > 0.95. Lowering the v_pred clip from the default
# 0.05 → 0.01 lets the sampler trust those predictions and refine more
# aggressively at very low noise — should tighten the slightly-underconfident
# contours if the residual MSE there is the dominant cause.
model.t_eps = 0.05
print(f"Sampler t_eps overridden to {model.t_eps}")

import pickle
import os

# Use the same MCMC reference the training run logged against. train_sharding.py
# defaults `mcmc_ref_dir` to "mcmc_log_normal", resolved from experiments/ — so
# from this notebook we go up one level.
mcmc_dir = "../mcmc_log_normal"
# mcmc_dir = "./mcmc_log_normal_sample/good_noise_level"

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

import jax_cosmo as jc
cosmo = jc.parameters.Planck15()  # used by the noiseless-truth block below

# NOTE: do NOT redraw a fresh y here. ``data['y']`` from the pickle is the exact
# observation the reference MCMC chain was conditioned on; conditioning JADE on
# anything else would compare two posteriors of two different observations.

from jade.sampling import EulerSampler, HeunSampler

# @jax.jit
def sample(obs, key, states=states, batch_size=128):
    nnx.update(model, states)
    #sampler = EulerSampler(model=model, num_steps=50)
    sampler = HeunSampler(model=model, num_steps=200)
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
samples_mcmc = MCSamples(samples=np.array(samples), names=names, label="MCMC")

theta_posterior = (cosmo_samples / SCALE_COSMO * THETA_STD + THETA_MEAN )
samples_posterior = MCSamples(samples=np.array(theta_posterior), names=names, label="JADE (our work)")

# Triangle plot (sometimes also called a corner plot)
# g = plots.get_subplot_plotter()
# g.triangle_plot([samples_posterior, samples_mcmc], names, markers=data['theta'], marker_args={"lw": 1}, filled=True)

# plt.savefig(os.path.join(save_dir, "contour_plot.png"))
# Triangle plot (sometimes also called a corner plot)
g = plots.get_subplot_plotter()

# Increase font sizes
g.settings.axes_fontsize = 34       # Tick labels
g.settings.axes_labelsize = 38      # Axis labels
g.settings.legend_fontsize = 24     # Legend text
g.settings.lab_fontsize = 38        # Label fontsize (getdist alias)
g.settings.tight_layout = True

g.triangle_plot(
    [samples_posterior, samples_mcmc],
    names,
    markers=data['theta'],
    marker_args={"lw": 1},
    filled=[True, False],
    line_args=[
        {"ls": "-", "color": "#d06e99ff"},  # JADE: solid line with your color
        {"ls": "--", "color": "black"}       # MCMC: dashed black line
    ],
    alpha=[1.,0.1],
    contour_colors=["#d06e99ff", "black"],  # Colors for 2D contours
    contour_ls=["-", "--"],  # Line styles for 2D contours
    contour_lws=[4., 4.]  # Line widths for 2D contours
)

# Bump tick label sizes on every subplot (getdist's axes_fontsize sometimes
# only updates the major numerical labels; this also catches minor ticks).
for ax_row in g.subplots:
    for ax_ in ax_row:
        if ax_ is not None:
            ax_.tick_params(axis='both', which='major', labelsize=28, length=8, width=1.2)
            ax_.tick_params(axis='both', which='minor', length=4, width=1.0)
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

# =============================================================================
# Persist the arrays needed for downstream analysis into ``save_dir``:
#   - posterior samples: JADE convergence-field draws (physical units) and the
#     matching cosmology draws (physical units).
#   - observation: the exact noisy convergence map the posterior conditions on.
#   - ground truth: the noiseless reference field + true cosmology (same objects
#     the joint-posterior figure labels "Ground Truth").
#   - Kaiser-Squires smoothed observation: since ``data['y']`` is already a
#     (noisy) convergence map, the KS mass map is a per-z-bin Gaussian smoothing
#     of the observation at a 2 arcmin scale.
# Saved here (right after data_no_noise is built) so the arrays land on disk even
# if any of the later plotting blocks fail.
# =============================================================================
from scipy.ndimage import gaussian_filter

_pix_arcmin = 5 * 60 / 128            # arcmin per pixel (5 deg field / 128 px)
_ks_smooth_arcmin = 2.0              # requested Gaussian smoothing scale
_ks_sigma_px = _ks_smooth_arcmin / _pix_arcmin

_obs_np = np.asarray(data["y"])                       # (128, 128, 5) noisy kappa
ks_smoothed_obs = np.stack(
    [gaussian_filter(_obs_np[..., c], sigma=_ks_sigma_px)
     for c in range(_obs_np.shape[-1])],
    axis=-1,
)                                                     # (128, 128, 5)

# posterior kappa in physical units (undo the field normalization)
posterior_kappa_samples = np.asarray(x_samples) * FIELD_STD + FIELD_MEAN
posterior_cosmo_samples = np.asarray(theta_posterior)  # physical units
ground_truth_field = np.asarray(data_no_noise["y"])
ground_truth_theta = np.asarray(data["theta"])

np.save(os.path.join(save_dir, "posterior_kappa_samples.npy"), posterior_kappa_samples)
np.save(os.path.join(save_dir, "posterior_cosmo_samples.npy"), posterior_cosmo_samples)
np.save(os.path.join(save_dir, "observation.npy"), _obs_np)
np.save(os.path.join(save_dir, "ground_truth_field.npy"), ground_truth_field)
np.save(os.path.join(save_dir, "ground_truth_theta.npy"), ground_truth_theta)
np.save(os.path.join(save_dir, "ks_smoothed_observation.npy"), ks_smoothed_obs)

# also bundle everything (plus the KS smoothing scale) into one archive
np.savez(
    os.path.join(save_dir, "amortized_arrays.npz"),
    posterior_kappa_samples=posterior_kappa_samples,
    posterior_cosmo_samples=posterior_cosmo_samples,
    observation=_obs_np,
    ground_truth_field=ground_truth_field,
    ground_truth_theta=ground_truth_theta,
    ks_smoothed_observation=ks_smoothed_obs,
    ks_smooth_arcmin=np.asarray(_ks_smooth_arcmin),
)
print(f"arrays saved to {save_dir}/ "
      f"(kappa samples {posterior_kappa_samples.shape}, "
      f"cosmo samples {posterior_cosmo_samples.shape})")

# =============================================================================
# Joint posterior figure: noisy observation, noiseless reference field, and
# JADE κ samples annotated with their inferred cosmology θ.
# Reuses the existing pipeline: ``data`` is the pickle the MCMC chain was run
# on (JADE conditions on the same y), ``data_no_noise`` is a Planck15 noiseless
# reference (statistically equivalent to the field underlying ``data['y']``),
# ``x_samples`` are the posterior κ draws, ``theta_posterior`` the matching
# cosmology in physical units. Two posterior samples are shown.
# =============================================================================
print("building joint posterior figure...")

_x_image = np.asarray(x_samples) * FIELD_STD + FIELD_MEAN
_n_instances = 2
_n_channels = 5
_cmap = "magma"
_tex_names = [r"\Omega_c", r"\Omega_b", r"\sigma_8", r"h_0", r"n_s", r"w_0"]

_obs = np.asarray(data["y"])
_ref = np.asarray(data_no_noise["y"])
_theta_truth = np.asarray(data["theta"])
_theta_post = np.asarray(theta_posterior)

_vmin_per_channel = [_ref[..., c].min() for c in range(_n_channels)]
_vmax_per_channel = [_ref[..., c].max() for c in range(_n_channels)]

fig_jp = plt.figure(figsize=(14, 2.5 * (_n_instances + 2)))
_gs = fig_jp.add_gridspec(
    _n_instances + 2,
    _n_channels + 2,
    width_ratios=[0.1, 0.6] + [1] * _n_channels,
    hspace=0.0,
    wspace=0.0,
)

_obs_axes = []
_last_sample_axes = []

# Row 0: noisy observation (own colorbar scale per channel)
_ax = fig_jp.add_subplot(_gs[0, 0]); _ax.axis("off")
_ax.text(0.5, 0.5, "Observation", fontsize=16, ha="center", va="center", rotation=90)
fig_jp.add_subplot(_gs[0, 1]).axis("off")
for _channel in range(_n_channels):
    _ax = fig_jp.add_subplot(_gs[0, _channel + 2])
    _im = _ax.imshow(_obs[..., _channel], cmap=_cmap)
    _ax.axis("off")
    _obs_axes.append((_ax, _im, _channel))

# Row 1: noiseless reference field at Planck15
_ax = fig_jp.add_subplot(_gs[1, 0]); _ax.axis("off")
_ax.text(0.5, 0.5, "Ground Truth", fontsize=16, ha="center", va="center", rotation=90)
_ax_text = fig_jp.add_subplot(_gs[1, 1]); _ax_text.axis("off")
_ax_text.text(
    0.5, 0.5,
    "\n".join(f"${_tex_names[i]}$: {float(_theta_truth[i]):.3f}" for i in range(6)),
    fontsize=14, ha="center", va="center",
)
for _channel in range(_n_channels):
    _ax = fig_jp.add_subplot(_gs[1, _channel + 2])
    _ax.imshow(_ref[..., _channel], cmap=_cmap,
               vmin=_vmin_per_channel[_channel], vmax=_vmax_per_channel[_channel])
    _ax.axis("off")

# Rows 2+: JADE posterior κ samples with their cosmology θ values
for _instance in range(_n_instances):
    _ax = fig_jp.add_subplot(_gs[_instance + 2, 0]); _ax.axis("off")
    _ax.text(0.5, 0.5, f"Sample {_instance + 1}", fontsize=16, ha="center", va="center", rotation=90)

    _ax_text = fig_jp.add_subplot(_gs[_instance + 2, 1]); _ax_text.axis("off")
    _ax_text.text(
        0.5, 0.5,
        "\n".join(f"${_tex_names[i]}$: {float(_theta_post[_instance, i]):.3f}" for i in range(6)),
        fontsize=14, ha="center", va="center",
    )
    for _channel in range(_n_channels):
        _ax = fig_jp.add_subplot(_gs[_instance + 2, _channel + 2])
        _im = _ax.imshow(_x_image[_instance, ..., _channel], cmap=_cmap,
                         vmin=_vmin_per_channel[_channel], vmax=_vmax_per_channel[_channel])
        _ax.axis("off")
        if _instance == _n_instances - 1:
            _last_sample_axes.append((_ax, _im))

# top colorbars (observation scale, one per channel)
for _ax, _im, _channel in _obs_axes:
    _pos = _ax.get_position()
    _cax = fig_jp.add_axes([_pos.x0, _pos.y1 + 0.002, _pos.width, 0.015])
    _cbar = plt.colorbar(_im, cax=_cax, orientation="horizontal")
    _cbar.ax.tick_params(labelsize=12)
    _cax.xaxis.set_ticks_position("top")
    _cax.xaxis.set_label_position("top")
    _cax.set_title(f"Bin {_channel}", fontsize=14, pad=10)

# bottom colorbars (κ-sample scale, one per channel)
for _ax, _im in _last_sample_axes:
    _pos = _ax.get_position()
    _cax = fig_jp.add_axes([_pos.x0, _pos.y0 - 0.017, _pos.width, 0.015])
    _cbar = plt.colorbar(_im, cax=_cax, orientation="horizontal")
    _cbar.ax.tick_params(labelsize=12)

plt.savefig(os.path.join(save_dir, "joint_posterior_samples.png"),
            bbox_inches="tight", pad_inches=0)
plt.savefig(os.path.join(save_dir, "joint_posterior_samples.pdf"),
            bbox_inches="tight", pad_inches=0)
plt.close(fig_jp)
print("joint posterior figure saved")

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

fontsize_text = 32  # For labels and legend
fontsize_ticks = 20  # For tick labels
fontsize_legend = 18
tick_length_major = 8
tick_length_minor = 4
tick_width = 1.4

fig, ax = plt.subplots(5,5,figsize=(10,10))
for i in range(5):
    for j in range(5):
        if j>i:
            ax[i, j].axis('off')
        else:
            if i==j:
                ax[i,j].loglog(ell, ps_auto[i], label='Ground truth', color="k", alpha=1.)
                ax[i,j].plot(ell, ps_auto_mean[i], color='tab:blue', alpha=1, label="JADE samples")
                ax[i,j].fill_between(ell, ps_auto_mean[i] - ps_auto_std[i],
                                     ps_auto_mean[i] + ps_auto_std[i],
                                     color='tab:blue', alpha=0.3)
                ax[i,j].set_xscale('log')
                ax[i,j].set_yscale('log')
            else:
                ax[i,j].loglog(ell, ps_cross[:, i, j], color='k')
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
            ax[i,j].tick_params(which='major', length=tick_length_major, width=tick_width)
            ax[i,j].tick_params(which='minor', length=tick_length_minor, width=tick_width)

        if i==4:
            ax[i,j].tick_params(axis='x', labelsize=fontsize_ticks)
        else:
            ax[i,j].tick_params(axis='x', labelbottom=False)

        if j==0:
            ax[i,j].tick_params(axis='y', labelsize=fontsize_ticks)
        else:
            ax[i,j].tick_params(axis='y', labelleft=False)

fig.supxlabel(r'$\ell$', fontsize=fontsize_text)
fig.supylabel(r'$\mathcal{C}_\ell$', fontsize=fontsize_text, x=-0.02)

# At the end, after the loops
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.88, 0.88), fontsize=fontsize_legend)
plt.savefig(os.path.join(save_dir,"power-spectra-posterior.png"), bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(save_dir,"power-spectra-posterior.pdf"), bbox_inches='tight', pad_inches=0.05)
plt.close(fig)


# Relative power spectrum: residuals against the C_l of the single noiseless
# convergence realization that produced the observed noisy data (the exact field
# the posterior is conditioned on reconstructing).
rel_auto_samples = (ps_auto[None] - ps_auto_samples) / ps_auto[None]

# avoid divide-by-zero on the diagonal of ps_cross (auto entries are 0 there)
ps_cross_for_rel = ps_cross[None]  # (1, n_ell, 5, 5)
_safe = np.where(ps_cross_for_rel == 0, 1.0, ps_cross_for_rel)
rel_cross_samples = (ps_cross_for_rel - ps_cross_samples) / _safe

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
                ax[i, j].axhline(0, color="k", alpha=1., label='Ground truth')
                ax[i, j].plot(ell, rel_auto_mean[i], color='tab:blue', alpha=1, label="JADE sample")
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
            ax[i, j].tick_params(which='major', length=tick_length_major, width=tick_width)
            ax[i, j].tick_params(which='minor', length=tick_length_minor, width=tick_width)

        if i == 4:
            ax[i, j].tick_params(axis='x', labelsize=fontsize_ticks)
        else:
            ax[i, j].tick_params(axis='x', labelbottom=False)

        if j == 0:
            ax[i, j].tick_params(axis='y', labelsize=fontsize_ticks)
        else:
            ax[i, j].tick_params(axis='y', labelleft=False)

fig.supxlabel(r'$\ell$', fontsize=fontsize_text)
fig.supylabel(r'$(\mathcal{C}_\ell^{\rm truth}-\mathcal{C}_\ell^{\rm sample})/\mathcal{C}_\ell^{\rm truth}$',
              fontsize=fontsize_text, x=-0.02)

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.88, 0.88), fontsize=fontsize_legend)
plt.savefig(os.path.join(save_dir, "power-spectra-posterior-relative.png"), bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(save_dir, "power-spectra-posterior-relative.pdf"), bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

# Average relative error per auto bin and overall.
# Compare the C_l of the single noiseless truth field to the posterior mean C_l,
# then take the absolute value and average over ell.
err_auto_means = (ps_auto - ps_auto_mean) / ps_auto
avg_rel_err_per_bin = np.mean(np.abs(err_auto_means), axis=1)  # shape (5,)
avg_rel_err_overall = np.mean(np.abs(err_auto_means))

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

    pdf_truth = gaussian_kde(truth_vals)(grid)

    pdf_samples = np.zeros((n_kde_samples, grid.size))
    for s in range(n_kde_samples):
        vals = np.asarray(x_samples_[s, :, :, i]).ravel()
        pdf_samples[s] = gaussian_kde(vals)(grid)
    pdf_mean = pdf_samples.mean(0)
    pdf_std = pdf_samples.std(0)

    ax[i].plot(grid, pdf_truth, color='k', label='Truth')
    ax[i].plot(grid, pdf_mean, color='tab:blue', label='JADE samples')
    ax[i].fill_between(grid, pdf_mean - pdf_std, pdf_mean + pdf_std,
                       color='tab:blue', alpha=0.3)
    ax[i].set_title(f'bin {i}', fontsize=fontsize_ticks)
    ax[i].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[i].xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[i].set_xlim(grid[0], grid[-1])

fig.supxlabel(r'$\kappa$', fontsize=fontsize_ticks, y=-0.08)
fig.supylabel(r'$p(\kappa)$', fontsize=fontsize_ticks)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=len(labels), fontsize=18, frameon=False)
plt.savefig(os.path.join(save_dir, "one-point-function.png"), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(save_dir, "one-point-function.pdf"), bbox_inches='tight', pad_inches=0)
plt.close(fig)


# Cross-correlation coefficient between the posterior MEAN field and the
# noiseless truth, per z-bin. r(l) = P_mt / sqrt(P_mm * P_tt).
x_mean_field = np.asarray(x_samples_).mean(axis=0)  # (128, 128, 5)
r_pmf = np.zeros((5, len(ell)))
for i in range(5):
    truth_map = ConvergenceMap(m_data[:, :, i], angle=map_size * u.deg)
    mean_map = ConvergenceMap(x_mean_field[:, :, i], angle=map_size * u.deg)
    _, p_mt = mean_map.cross(truth_map, l_edges=l_edges_kmap)
    _, p_mm = mean_map.cross(mean_map, l_edges=l_edges_kmap)
    p_tt = ps_auto[i]
    r_pmf[i] = p_mt / np.sqrt(p_mm * p_tt)

fig, ax = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
for i in range(5):
    ax[i].axhline(1.0, color='k', label='Perfect correlation')
    ax[i].plot(ell, r_pmf[i], color='tab:blue', label='Posterior mean field')
    ax[i].set_xscale('log')
    ax[i].set_xlim(ell.min(), ell.max())
    ax[i].set_title(f'bin {i + 1}', fontsize=fontsize_text)
    ax[i].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[i].xaxis.set_minor_formatter(NullFormatter())

fig.supxlabel(r'$\ell$', fontsize=fontsize_text)
fig.supylabel(r'$r(\ell)$', fontsize=fontsize_text)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=len(labels), fontsize=18, frameon=False)
plt.savefig(os.path.join(save_dir, "cross-correlation-coefficient.png"), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(save_dir, "cross-correlation-coefficient.pdf"), bbox_inches='tight', pad_inches=0)
plt.close(fig)

