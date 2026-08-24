"""Cosmic-variance-cancelled power-spectrum comparison: JADE vs ground truth.

This is the averaged counterpart of the single-realization comparison in
``plot_amortized.py`` (see ``power-spectra-truth-vs-jade.png``). Instead of
conditioning on one observation, we:

  1. Draw ``N_REAL`` (=10) lensing convergence fields at a FIXED fiducial
     cosmology (Planck15 -- the same theta the reference MCMC chain used).
  2. For each field keep the noiseless map (ground truth) AND add LSST-Y10 shape
     noise to it (matched pair) to build the JADE observation.
  3. Sample JADE posterior kappa maps conditioned on each noisy observation.
  4. Average the ground-truth C_ell over the 10 fields (cancels cosmic variance)
     and average the JADE-sample C_ell over all pooled posterior maps.
  5. Plot the 5x5 lower-triangle (auto on the diagonal, cross off-diagonal).

Truth = black line; JADE = blue mean +/- 1 sigma sample-scatter band.
"""

import os
import itertools

import jax
import jax.numpy as jnp

from flax import nnx

import matplotlib as mpl
import matplotlib.pyplot as plt
# Paper-style typography without a system TeX install (see plot_amortized.py).
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})
import numpy as np
from tqdm import tqdm

import astropy.units as u
from lenstools import ConvergenceMap

from jade.nn_hybrid import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, sigma_lsst
from jade.flow import Denoiser
from jade.utils import load_model
from jade.sampling import HeunSampler

print(jax.devices())
print("sigma_lsst", sigma_lsst)

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
N_REAL = int(os.environ.get("N_REAL", 100))              # noisy observations fed to JADE
N_TRUTH = int(os.environ.get("N_TRUTH", 100))            # noiseless fields averaged for the truth C_ell
N_SAMPLES_PER_OBS = int(os.environ.get("N_SAMPLES_PER_OBS", 10))  # JADE kappa maps per observation
# Posterior sampling is vmapped over ALL (obs, sample) pairs at once and run in
# chunks of N_CHUNK to bound GPU memory (one compile, reused across chunks).
N_CHUNK = int(os.environ.get("N_CHUNK", 100))
N = 128
map_size = 5             # deg
save_dir = "amortized_cosmic_variance"
os.makedirs(save_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load JADE (same run / settings as plot_amortized.py)
# ---------------------------------------------------------------------------
PARAMS_NAME = os.environ.get("PARAMS_NAME", "JADE_B_16_ema_best")
print("checkpoint params:", PARAMS_NAME)
cfg, states = load_model(
    "/u/bremy/repos/jade/experiments/wandb/run-20260507_170014-7hnur00g/files/checkpoints",
    PARAMS_NAME,
)

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


def sample_batched(conds, key, chunk=N_CHUNK):
    """Sample one posterior kappa map per row of `conds` (M, 128, 128, 5).

    The sampler is vmapped over BOTH the initial noise AND the conditioning
    (in_axes=(0,0,0,0)), so distinct observations are denoised in parallel.
    Work is split into chunks of `chunk` rows to bound GPU memory; all chunks
    share one compiled vmap (pad the last chunk if M is not a multiple).
    """
    nnx.update(model, states)
    sampler = HeunSampler(model=model, num_steps=200)
    M = conds.shape[0]
    k0, k1, k2 = jax.random.split(key, 3)
    x_0_all = jax.random.normal(k0, shape=(M, 128, 128, 5))
    cosmo_0_all = jax.random.normal(k1, shape=(M, 6))
    keys_all = jax.random.split(k2, M)
    cond_all = (conds - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)

    vmapped = jax.vmap(sampler, in_axes=(0, 0, 0, 0))
    x_out, cosmo_out = [], []
    for s in tqdm(range(0, M, chunk), desc="JADE sampling"):
        e = min(s + chunk, M)
        x, c = vmapped(x_0_all[s:e], cosmo_0_all[s:e], cond_all[s:e], keys_all[s:e])
        x_out.append(np.asarray(x))
        cosmo_out.append(np.asarray(c))
    return np.concatenate(x_out), np.concatenate(cosmo_out)


# ---------------------------------------------------------------------------
# Generate 10 (clean, noisy) realizations at the FIXED fiducial cosmology.
# Only the field realization (z) varies between realizations; cosmology is held.
# ---------------------------------------------------------------------------
import jax_cosmo as jc
from numpyro.handlers import condition, seed, trace
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from functools import partial

cosmo = jc.parameters.Planck15()  # fiducial; equals data['theta'] of the MCMC ref

model_log_normal = partial(
    lensingLogNormal,
    N=N,
    map_size=map_size,
    gal_per_arcmin2=config_lsst_y_10.gals_per_arcmin2,
    sigma_e=config_lsst_y_10.sigma_e,
    nbins=config_lsst_y_10.nbins,
    a=config_lsst_y_10.a,
    b=config_lsst_y_10.b,
    z0=config_lsst_y_10.z0,
    model_type='lognormal',
    lognormal_shifts='LSSTY10',
    with_noise=False,   # we add matched noise ourselves below
)

fiducial = {
    "omega_c": cosmo.Omega_c,
    "omega_b": cosmo.Omega_b,
    "sigma_8": cosmo.sigma8,
    "h_0": cosmo.h,
    "n_s": cosmo.n_s,
    "w_0": cosmo.w0,
}


def draw_clean_field(key):
    """Noiseless convergence field (128,128,5) at the fiducial cosmology."""
    m = seed(model_log_normal, key)
    m = condition(m, fiducial)
    tr = trace(m).get_trace()
    return jnp.asarray(tr["y"]["value"])


key = jax.random.key(0)
clean_fields = []      # ground-truth maps (N_TRUTH of them, for the truth-mean C_ell)
noisy_obs = []         # JADE observations: clean + LSST noise, only the first N_REAL fields
for i in tqdm(range(N_TRUTH), desc="drawing fields"):
    key, k_field, k_noise = jax.random.split(key, 3)
    clean = draw_clean_field(k_field)
    clean_fields.append(np.asarray(clean))
    if i < N_REAL:
        noise = jax.random.normal(k_noise, shape=clean.shape) * sigma_lsst.reshape(1, 1, -1)
        noisy_obs.append(np.asarray(clean + noise))
clean_fields = np.stack(clean_fields)   # (N_TRUTH, 128, 128, 5)
noisy_obs = np.stack(noisy_obs)         # (N_REAL, 128, 128, 5)

# ---------------------------------------------------------------------------
# JADE posterior sampling: all (obs, sample) pairs vmapped together, chunked.
# Each observation is repeated N_SAMPLES_PER_OBS times in the conditioning stack.
# ---------------------------------------------------------------------------
cond_stack = np.repeat(noisy_obs, N_SAMPLES_PER_OBS, axis=0)  # (N_REAL*N_SAMPLES, 128,128,5)
key, subkey = jax.random.split(key)
x_samples, _ = sample_batched(jnp.asarray(cond_stack), subkey)
all_post_maps = np.asarray(x_samples) * FIELD_STD + FIELD_MEAN  # (N_REAL*N_SAMPLES, 128,128,5)
print("pooled posterior maps:", all_post_maps.shape)

# ---------------------------------------------------------------------------
# Power spectra (auto + cross), reusing the plot_amortized.py machinery.
# ---------------------------------------------------------------------------
l_edges_kmap = np.linspace(500, 4608.0, 128)


def fill_lower_diag(array, nl):
    n = int(np.sqrt(len(array) * 2)) + 1
    mask = np.arange(n)[:, None] > np.arange(n)
    out = np.zeros((n, n, nl))
    out[np.stack(mask, axis=1)] = array
    return out.T


def compute_ps(m_data1, m_data2):
    lis = [0, 1, 2, 3, 4]
    p_cross = []
    for i, j in itertools.combinations(lis, 2):
        ell, ps = ConvergenceMap(m_data1[:, :, i], angle=map_size * u.deg).cross(
            ConvergenceMap(m_data2[:, :, j], angle=map_size * u.deg),
            l_edges=l_edges_kmap,
        )
        p_cross.append(ps)
    ps_cross = fill_lower_diag(np.array(p_cross), 127)

    ps_auto = []
    for i in range(5):
        ell, pi = ConvergenceMap(m_data1[:, :, i], angle=map_size * u.deg).cross(
            ConvergenceMap(m_data2[:, :, i], angle=map_size * u.deg),
            l_edges=l_edges_kmap,
        )
        ps_auto.append(pi)
    return ell, np.array(ps_auto), ps_cross


# Ground truth: average C_ell over the N_TRUTH noiseless fields (cancels cosmic var).
truth_auto = []
truth_cross = []
for i in tqdm(range(N_TRUTH), desc="truth ps"):
    ell, a, c = compute_ps(clean_fields[i], clean_fields[i])
    truth_auto.append(a)
    truth_cross.append(c)
truth_auto = np.array(truth_auto)
truth_cross = np.array(truth_cross)
ps_auto_truth = truth_auto.mean(0)
ps_cross_truth = truth_cross.mean(0)

# JADE: C_ell for every pooled posterior map; mean +/- sample-scatter std.
post_auto = []
post_cross = []
for s in tqdm(range(len(all_post_maps)), desc="jade ps"):
    ell, a, c = compute_ps(all_post_maps[s], all_post_maps[s])
    post_auto.append(a)
    post_cross.append(c)
post_auto = np.array(post_auto)
post_cross = np.array(post_cross)
ps_auto_mean = post_auto.mean(0)
ps_auto_std = post_auto.std(0)
ps_cross_mean = post_cross.mean(0)
ps_cross_std = post_cross.std(0)

# ---------------------------------------------------------------------------
# Cache the expensive products (drawn fields, JADE posterior maps, all
# per-realization power spectra) ONCE, so the chi-square diagnostics can be
# iterated without re-running the GPU sampling. Analyze with chi2_cv.py.
# ---------------------------------------------------------------------------
CACHE_TAG = os.environ.get(
    "CACHE_TAG", f"N{N_REAL}x{N_SAMPLES_PER_OBS}_T{N_TRUTH}_{PARAMS_NAME}"
)
CACHE_PATH = os.path.join(save_dir, f"cv_cache_{CACHE_TAG}.npz")
np.savez(
    CACHE_PATH,
    clean_fields=clean_fields,
    noisy_obs=noisy_obs,
    all_post_maps=all_post_maps,
    ell=ell,
    truth_auto=truth_auto,
    truth_cross=truth_cross,
    post_auto=post_auto,
    post_cross=post_cross,
    sigma_lsst=np.asarray(sigma_lsst),
    N_REAL=N_REAL,
    N_SAMPLES_PER_OBS=N_SAMPLES_PER_OBS,
    N_TRUTH=N_TRUTH,
    PARAMS_NAME=PARAMS_NAME,
)
print(f"cached samples + spectra -> {CACHE_PATH}")

# ---------------------------------------------------------------------------
# Plot: 5x5 lower-triangle (auto on diagonal, cross off-diagonal).
# ---------------------------------------------------------------------------
from matplotlib.ticker import LogLocator, NullFormatter

fontsize_text = 32
fontsize_ticks = 20
fontsize_legend = 18
tick_length_major = 8
tick_length_minor = 4
tick_width = 1.4

fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        if j > i:
            ax[i, j].axis('off')
        else:
            if i == j:
                ax[i, j].loglog(ell, ps_auto_truth[i], color="k", alpha=1.,
                                label=r'Truth $\kappa$')
                ax[i, j].plot(ell, ps_auto_mean[i], color='tab:blue', alpha=1.,
                              label="JADE samples")
                ax[i, j].fill_between(ell, ps_auto_mean[i] - ps_auto_std[i],
                                      ps_auto_mean[i] + ps_auto_std[i],
                                      color='tab:blue', alpha=0.3)
            else:
                ax[i, j].loglog(ell, ps_cross_truth[:, i, j], color='k')
                ax[i, j].plot(ell, ps_cross_mean[:, i, j], color='tab:blue', alpha=1.)
                ax[i, j].fill_between(ell, ps_cross_mean[:, i, j] - ps_cross_std[:, i, j],
                                      ps_cross_mean[:, i, j] + ps_cross_std[:, i, j],
                                      color='tab:blue', alpha=0.3)
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
plt.savefig(os.path.join(save_dir, "power-spectra-truth-vs-jade-avg.png"),
            bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(save_dir, "power-spectra-truth-vs-jade-avg.pdf"),
            bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print("saved power-spectra-truth-vs-jade-avg.{png,pdf} to", save_dir)

# Power-spectrum + pixel chi-square diagnostics now live in chi2_cv.py, which
# loads the cache written above -- no GPU / re-sampling needed. Run e.g.:
#   python chi2_cv.py
#   CACHE_TAG=N100x10_T100_JADE_B_16_ema_best python chi2_cv.py
