"""Validation script analogous to plot_amortized.py.

Differences:
  * The noisy observation and the noiseless reference field share the *same*
    initial condition. Both are generated at Planck15 with the same seed,
    and the noisy map is constructed explicitly as noiseless + shape noise.
  * Two figures are produced:
      1. ``power-spectra-posterior.{png,pdf}`` — 5×5 triangle plot of auto +
         cross power spectra of the noiseless truth against 20 individual
         posterior samples, plus the posterior mean ± 1σ band.
      2. ``cross-correlation-coefficient.{png,pdf}`` — bin-i vs bin-i cross
         correlation coefficient r(ℓ) between each of 20 posterior samples
         and the noiseless truth, plus the mean ± 1σ band.
"""

import os

import jax
import jax.numpy as jnp
from flax import nnx

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})
import numpy as np

from jade.nn_hybrid import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.flow import Denoiser
from jade.utils import load_model
from jade.sampling import HeunSampler

print(jax.devices())

save_dir = 'amortized_validation'
os.makedirs(save_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the same checkpoint as plot_amortized.py.
# ---------------------------------------------------------------------------
cfg, states = load_model(
    "/u/bremy/repos/jade/experiments/wandb/run-20260507_170014-7hnur00g/files/checkpoints",
    "JADE_B_16_latest",
)

SCALE_COSMO = 1

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
    split_qkv=cfg['model'].get('split_qkv', False),
    mask_theta_to_field=cfg['model'].get('mask_theta_to_field', False),
)

model = Denoiser(model, cfg)
nnx.update(model, states)
model.t_eps = 0.05
print(f"Sampler t_eps overridden to {model.t_eps}")

# ---------------------------------------------------------------------------
# Generate noiseless + noisy maps at Planck15 with shared initial conditions.
# We draw the noiseless field once with seed K, then sample shape noise from
# the model's noise distribution and add it to obtain the matching noisy map.
# ---------------------------------------------------------------------------
import jax_cosmo as jc
from numpyro.handlers import condition, seed, trace
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from sbi_lens.simulator.redshift import subdivide
from functools import partial

cosmo = jc.parameters.Planck15()

sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b = config_lsst_y_10.b
z0 = config_lsst_y_10.z0
N = 128
map_size = 5  # degrees

# Per-bin galaxy density (used to build the shape-noise covariance).
nz_total = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gals_per_arcmin2)
nz_bins = subdivide(nz_total, nbins=nbins, zphot_sigma=0.05)
gals_per_arcmin2_per_bin = jnp.array([nzb.gals_per_arcmin2 for nzb in nz_bins])
pix_area_arcmin2 = (map_size * 60 / N) ** 2
sigma_noise_per_bin = sigma_e / jnp.sqrt(gals_per_arcmin2_per_bin * pix_area_arcmin2)
print("per-bin shape-noise σ:", np.asarray(sigma_noise_per_bin))

model_no_noise = partial(
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
    with_noise=False,
)

KEY_IC = jax.random.key(0)
KEY_NOISE = jax.random.key(1)

cond_model = seed(model_no_noise, KEY_IC)
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
truth_trace = trace(cond_model).get_trace()
theta_truth = jnp.stack([truth_trace[name]["value"] for name in params_name], axis=-1)
kappa_truth = truth_trace["y"]["value"]  # (N, N, nbins), noiseless

# Shape noise: independent per pixel per bin, σ_i per bin from the model spec.
noise = jax.random.normal(KEY_NOISE, shape=kappa_truth.shape) * sigma_noise_per_bin.reshape(1, 1, -1)
y_noisy = kappa_truth + noise

data = {"theta": theta_truth, "y": y_noisy}
data_no_noise = {"theta": theta_truth, "y": kappa_truth}

print("noisy − noiseless std (per bin):",
      np.asarray(jnp.std(y_noisy - kappa_truth, axis=(0, 1))))
print("expected per-bin σ:", np.asarray(sigma_noise_per_bin))

# Quick observation visualisation.
plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.asarray(y_noisy[..., i]))
    plt.title(f"noisy bin {i}")
    plt.colorbar()
    plt.subplot(2, 5, i + 6)
    plt.imshow(np.asarray(kappa_truth[..., i]))
    plt.title(f"noiseless bin {i}")
    plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "observation.png"), bbox_inches="tight")
plt.close()
print("observation figure saved")

# ---------------------------------------------------------------------------
# Sample the full JADE posterior conditioned on the noisy observation.
# ---------------------------------------------------------------------------
def sample(obs, key, states=states, batch_size=128):
    nnx.update(model, states)
    sampler = HeunSampler(model=model, num_steps=200)
    keys = jax.random.split(key, 3)
    x_0 = jax.random.normal(keys[0], shape=(batch_size, 128, 128, 5))
    cosmo_0 = jax.random.normal(keys[1], shape=(batch_size, 6))
    keys = jax.random.split(keys[2], batch_size)
    cond = (obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)
    x_samples, cosmo_samples = jax.vmap(sampler, in_axes=(0, 0, None, 0))(
        x_0, cosmo_0, cond, keys
    )
    return x_samples, cosmo_samples


from tqdm import tqdm

obs = data['y']
key = jax.random.key(42)
x_samples, cosmo_samples = sample(obs, key, states)
for _ in tqdm(range(3), desc="posterior batches"):
    key, subkey = jax.random.split(key)
    x_samples_, cosmo_samples_ = sample(obs, subkey, states)
    x_samples = jnp.concatenate([x_samples, x_samples_])
    cosmo_samples = jnp.concatenate([cosmo_samples, cosmo_samples_])

theta_posterior = cosmo_samples / SCALE_COSMO * THETA_STD + THETA_MEAN
print(f"posterior samples drawn: {x_samples.shape[0]}")

# Cosmology triangle plot (sanity, matches plot_amortized.py style).
from getdist import MCSamples, plots
names = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]
samples_posterior = MCSamples(
    samples=np.array(theta_posterior), names=names, label="JADE (our work)"
)
g = plots.get_subplot_plotter()
g.settings.axes_fontsize = 28
g.settings.axes_labelsize = 32
g.settings.legend_fontsize = 22
g.settings.lab_fontsize = 32
g.settings.tight_layout = True
g.triangle_plot(
    [samples_posterior], names,
    markers=np.asarray(theta_truth),
    marker_args={"lw": 1},
    filled=[True],
    line_args=[{"ls": "-", "color": "#d06e99ff"}],
    contour_colors=["#d06e99ff"],
    contour_ls=["-"],
    contour_lws=[3.0],
)
for ax_row in g.subplots:
    for ax_ in ax_row:
        if ax_ is not None:
            ax_.tick_params(axis='both', which='major', labelsize=22, length=7, width=1.0)
            ax_.tick_params(axis='both', which='minor', length=3, width=0.8)
plt.savefig(os.path.join(save_dir, "contour_plot.png"))
plt.savefig(os.path.join(save_dir, "contour_plot.pdf"))
plt.close()
print("contour plot saved")

# ---------------------------------------------------------------------------
# Power spectra (auto + cross between z-bins).
# ---------------------------------------------------------------------------
import astropy.units as u
from lenstools import ConvergenceMap
import itertools

l_edges_kmap = np.linspace(500, 4608.0, 128)


def fill_lower_diag(array, nl):
    n = int(np.sqrt(len(array) * 2)) + 1
    mask = np.arange(n)[:, None] > np.arange(n)
    out = np.zeros((n, n, nl))
    out[np.stack(mask, axis=1)] = array
    return out.T


def compute_ps(m_data1, m_data_2):
    lis = [0, 1, 2, 3, 4]
    p_cross = []
    for i, j in itertools.combinations(lis, 2):
        ell, ps = ConvergenceMap(
            m_data1[:, :, i], angle=map_size * u.deg
        ).cross(
            ConvergenceMap(m_data_2[:, :, j], angle=map_size * u.deg),
            l_edges=l_edges_kmap,
        )
        p_cross.append(ps)
    ps_cross = np.array(p_cross)
    ps_cross = fill_lower_diag(ps_cross, 127)
    ps_auto = []
    for i in range(5):
        ell, pi = ConvergenceMap(
            m_data1[:, :, i], angle=map_size * u.deg
        ).cross(
            ConvergenceMap(m_data_2[:, :, i], angle=map_size * u.deg),
            l_edges=l_edges_kmap,
        )
        ps_auto.append(pi)
    ps_auto = np.array(ps_auto)
    return ell, ps_auto, ps_cross


# Rescale posterior samples back to physical κ units.
x_samples_phys = np.asarray(x_samples) * np.asarray(FIELD_STD) + np.asarray(FIELD_MEAN)

m_truth = np.asarray(kappa_truth)
ell, ps_truth_auto, ps_truth_cross = compute_ps(m_truth, m_truth)

# Restrict to the first 20 posterior samples for both figures.
N_PLOT_SAMPLES = 20
selected = x_samples_phys[:N_PLOT_SAMPLES]

ps_auto_samples = []
ps_cross_samples = []
for s in tqdm(range(N_PLOT_SAMPLES), desc="sample power spectra"):
    _, ps_a, ps_c = compute_ps(selected[s], selected[s])
    ps_auto_samples.append(ps_a)
    ps_cross_samples.append(ps_c)
ps_auto_samples = np.array(ps_auto_samples)        # (S, 5, n_ell)
ps_cross_samples = np.array(ps_cross_samples)      # (S, n_ell, 5, 5)

ps_auto_mean = ps_auto_samples.mean(0)
ps_auto_std = ps_auto_samples.std(0)
ps_cross_mean = ps_cross_samples.mean(0)
ps_cross_std = ps_cross_samples.std(0)

# ---------------------------------------------------------------------------
# Figure 1: triangle plot of auto + cross power spectra.
# ---------------------------------------------------------------------------
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

fontsize_text = 32
fontsize_ticks = 20
fontsize_legend = 18
tick_length_major = 8
tick_length_minor = 4
tick_width = 1.4

sample_color = "tab:blue"
sample_line_alpha = 0.25

fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        if j > i:
            ax[i, j].axis('off')
            continue
        if i == j:
            ax[i, j].loglog(ell, ps_truth_auto[i], color='k', alpha=1.0, lw=2,
                            label='Noiseless truth')
            for s in range(N_PLOT_SAMPLES):
                ax[i, j].plot(ell, ps_auto_samples[s, i], color=sample_color,
                              alpha=sample_line_alpha, lw=1,
                              label='JADE samples' if s == 0 else None)
            ax[i, j].plot(ell, ps_auto_mean[i], color=sample_color, alpha=1.0, lw=2,
                          label='Posterior mean')
            ax[i, j].fill_between(ell, ps_auto_mean[i] - ps_auto_std[i],
                                  ps_auto_mean[i] + ps_auto_std[i],
                                  color=sample_color, alpha=0.3)
        else:
            ax[i, j].loglog(ell, ps_truth_cross[:, i, j], color='k', lw=2)
            for s in range(N_PLOT_SAMPLES):
                ax[i, j].plot(ell, ps_cross_samples[s, :, i, j],
                              color=sample_color, alpha=sample_line_alpha, lw=1)
            ax[i, j].plot(ell, ps_cross_mean[:, i, j], color=sample_color,
                          alpha=1.0, lw=2)
            ax[i, j].fill_between(ell, ps_cross_mean[:, i, j] - ps_cross_std[:, i, j],
                                  ps_cross_mean[:, i, j] + ps_cross_std[:, i, j],
                                  color=sample_color, alpha=0.3)
        ax[i, j].set_xscale('log')
        ax[i, j].set_yscale('log')
        ax[i, j].set_xlim(ell.min(), ell.max())
        ax[i, j].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax[i, j].yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        ax[i, j].xaxis.set_minor_formatter(NullFormatter())
        ax[i, j].yaxis.set_minor_formatter(NullFormatter())
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
fig.supylabel(r'$\mathcal{C}_\ell$', fontsize=fontsize_text, x=-0.02)
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.88, 0.88),
           fontsize=fontsize_legend)
plt.savefig(os.path.join(save_dir, "power-spectra-posterior.png"),
            bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(save_dir, "power-spectra-posterior.pdf"),
            bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print("power-spectra triangle figure saved")

# ---------------------------------------------------------------------------
# Figure 2: bin-i vs bin-i cross-correlation coefficient between each
# posterior sample and the noiseless truth.
#
# r_i(ℓ) = C_{s_i, t_i} / sqrt(C_{s_i, s_i} * C_{t_i, t_i})
# ---------------------------------------------------------------------------
r_samples = np.zeros((N_PLOT_SAMPLES, 5, len(ell)))
for s in tqdm(range(N_PLOT_SAMPLES), desc="cross-correlation coefficient"):
    for i in range(5):
        truth_map = ConvergenceMap(m_truth[:, :, i], angle=map_size * u.deg)
        sample_map = ConvergenceMap(selected[s, :, :, i], angle=map_size * u.deg)
        _, p_st = sample_map.cross(truth_map, l_edges=l_edges_kmap)
        p_ss = ps_auto_samples[s, i]
        p_tt = ps_truth_auto[i]
        r_samples[s, i] = p_st / np.sqrt(p_ss * p_tt)

r_mean = r_samples.mean(0)
r_std = r_samples.std(0)

fig, ax = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
for i in range(5):
    ax[i].axhline(1.0, color='k', lw=2, label='Perfect correlation')
    for s in range(N_PLOT_SAMPLES):
        ax[i].plot(ell, r_samples[s, i], color=sample_color,
                   alpha=sample_line_alpha, lw=1,
                   label='JADE samples' if (s == 0 and i == 0) else None)
    ax[i].plot(ell, r_mean[i], color=sample_color, lw=2,
               label='Posterior mean' if i == 0 else None)
    ax[i].fill_between(ell, r_mean[i] - r_std[i], r_mean[i] + r_std[i],
                       color=sample_color, alpha=0.3)
    ax[i].set_xscale('log')
    ax[i].set_xlim(ell.min(), ell.max())
    ax[i].set_ylim(0.0, 1.05)
    ax[i].set_title(f'bin {i + 1}', fontsize=fontsize_text)
    ax[i].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[i].xaxis.set_minor_formatter(NullFormatter())
    ax[i].tick_params(which='major', length=tick_length_major, width=tick_width)
    ax[i].tick_params(which='minor', length=tick_length_minor, width=tick_width)

fig.supxlabel(r'$\ell$', fontsize=fontsize_text)
fig.supylabel(r'$r(\ell)$', fontsize=fontsize_text)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=len(labels), fontsize=fontsize_legend, frameon=False)
plt.savefig(os.path.join(save_dir, "cross-correlation-coefficient.png"),
            bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(save_dir, "cross-correlation-coefficient.pdf"),
            bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print("cross-correlation coefficient figure saved")
