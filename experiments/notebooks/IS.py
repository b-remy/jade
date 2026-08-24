import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import partial
from tqdm import tqdm

from flax import nnx
from getdist import MCSamples, plots

from jade.nn_hybrid import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, GRF_MEAN, GRF_STD
from jade.flow import Denoiser
from jade.utils import load_model
from jade.sampling import EulerSampler

import jax_cosmo as jc
from numpyro.handlers import condition, seed, trace
from numpyro.infer.util import log_density
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from sbi_lens.simulator.LogNormal_field import (
    shift_fn, fill_shift_array, make_power_map, make_lognormal_power_map, DATA_DIR
)

print(jax.devices())

# ============================================================
# Config
# ============================================================
save_dir = 'IS'
os.makedirs(save_dir, exist_ok=True)

SCALE_COSMO = 1
N = 128
map_size = 5
sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b_nz = config_lsst_y_10.b
z0 = config_lsst_y_10.z0

params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]
names = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]

# ============================================================
# Load model (trained on g-space)
# ============================================================
cfg, states = load_model(
    "/u/bremy/repos/jade/experiments/wandb/run-20260226_093505-3gytbpbd/files/checkpoints",
    "JADE_B_16_ema_latest",
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
)
model = Denoiser(model, cfg)
nnx.update(model, states)

# ============================================================
# Load observation and MCMC samples
# ============================================================
mcmc_dir = "./mcmc_log_normal_sample/good_noise_level"
with open(os.path.join(mcmc_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
    obs_truth = pickle.load(f)
with open(os.path.join(mcmc_dir, "mcmc_log_posterior_samples.pkl"), "rb") as f:
    mcmc_samples = pickle.load(f)

# ============================================================
# Generate observation (matching amortized script)
# ============================================================
import numpyro
import numpyro.distributions as dist

cosmo = jc.parameters.Planck15()
key = jax.random.key(0)

model_log_normal = partial(
    lensingLogNormal,
    N=N, map_size=map_size, gal_per_arcmin2=gals_per_arcmin2,
    sigma_e=sigma_e, nbins=nbins, a=a, b=b_nz, z0=z0,
    model_type='lognormal', lognormal_shifts='LSSTY10', with_noise=True,
)

cond_model_sim = seed(model_log_normal, key)
cond_model_sim = condition(
    cond_model_sim,
    {
        "omega_c": cosmo.Omega_c, "omega_b": cosmo.Omega_b,
        "sigma_8": cosmo.sigma8, "h_0": cosmo.h,
        "n_s": cosmo.n_s, "w_0": cosmo.w0,
    },
)

model_trace = trace(cond_model_sim).get_trace()
data = {
    "theta": jnp.stack([model_trace[name]["value"] for name in params_name], axis=-1),
    "y": model_trace["y"]["value"],
}

obs = data['y']
cond = (obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)

plt.figure()
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(obs[..., i])
    plt.colorbar()
plt.savefig(os.path.join(save_dir, "observation.png"))
print("obs saved")

# ============================================================
# Forward model config for IS
# ============================================================
fwd_kwargs = dict(
    N=N, nbins=nbins, map_size=map_size,
    gal_per_arcmin2=gals_per_arcmin2, sigma_e=sigma_e,
    a=a, b=b_nz, z0=z0, lognormal_shifts='LSSTY10',
)

conditioned_model = condition(model_log_normal, data={"y": obs})

def log_prob_z_fn(params):
    """log p(theta, z, y=obs) from numpyro"""
    return log_density(conditioned_model, (), {}, params)[0]

# Rescaling constants
log_det_rescale_g = -N * N * jnp.sum(jnp.log(GRF_STD))
log_det_rescale_theta = -jnp.sum(jnp.log(THETA_STD / SCALE_COSMO))

# ============================================================
# g -> z inversion and Jacobian (linear, exact)
# ============================================================
from jade.log_prob import _build_L_and_shift

def invert_g_to_z(g, theta, **kwargs):
    """
    Invert g -> z via L^{-1} in Fourier space.
    g has shape (nbins, N, N).
    Returns z (nbins, N, N), log|det dg/dz| (scalar), and shift.
    """
    L_pix, eigval, shift = _build_L_and_shift(theta, **kwargs)
    L_fwd = L_pix.transpose([2, 3, 0, 1])  # (nbins_out, nbins_in, N, N)

    g_fft = jnp.fft.fft2(g)
    g_fft = g_fft.at[:, 0, 0].set(0.0)

    L_fourier = L_fwd.transpose([2, 3, 0, 1])  # (N, N, nbins, nbins)
    g_fft_pix = g_fft.transpose([1, 2, 0])      # (N, N, nbins)

    fft_z = jnp.linalg.solve(L_fourier[1:], g_fft_pix[1:][..., None]).squeeze(-1)
    fft_z_dc = jnp.zeros((1, N, nbins), dtype=fft_z.dtype)
    fft_z_full = jnp.concatenate([fft_z_dc, fft_z], axis=0)
    fft_z_full = fft_z_full.transpose([2, 0, 1])

    z = jnp.fft.ifft2(fft_z_full).real

    eigval_flat = eigval.reshape(N * N, nbins)[1:]
    log_det_L = 0.5 * jnp.sum(jnp.log(jnp.clip(eigval_flat, a_min=1e-30)))

    return z, log_det_L, shift


@jax.jit
def compute_log_p_g_and_components(g, theta):
    """
    Compute log p(g, theta | y) and its decomposition.
    Returns: log_p_g, log_pz_joint, log_p_z_prior, log_det_L
    """
    z, log_det_L, shift = invert_g_to_z(g, theta, **fwd_kwargs)

    omega_c, omega_b, sigma_8, h_0, n_s, w_0 = theta
    params = {
        "omega_c": omega_c, "omega_b": omega_b, "sigma_8": sigma_8,
        "h_0": h_0, "n_s": n_s, "w_0": w_0, "z": z,
    }
    log_pz_joint = log_prob_z_fn(params)
    log_p_z_prior = -0.5 * jnp.sum(z**2) - 0.5 * z.size * jnp.log(2*jnp.pi)
    log_p_g = log_pz_joint - log_det_L

    return log_p_g, log_pz_joint, log_p_z_prior, log_det_L


# ============================================================
# Sampling
# ============================================================
print("\n" + "=" * 60)
print("Sampling")
print("=" * 60)

# --- Euler samples (reference) ---
def sample_euler(key, batch_size=64):
    nnx.update(model, states)
    euler = EulerSampler(model=model, num_steps=128)
    keys = jax.random.split(key, 3)
    g_0 = jax.random.normal(keys[0], shape=(batch_size, N, N, nbins))
    cosmo_0 = jax.random.normal(keys[1], shape=(batch_size, 6))
    sample_keys = jax.random.split(keys[2], batch_size)
    g_s, c_s = jax.vmap(euler, in_axes=(0, 0, None, 0))(g_0, cosmo_0, cond, sample_keys)
    return g_s, c_s

key = jax.random.key(0)
g_euler_all, cosmo_euler_all = sample_euler(key)
for i in tqdm(range(3), desc="Euler batches"):
    key, subkey = jax.random.split(key)
    g_s, c_s = sample_euler(subkey)
    g_euler_all = jnp.concatenate([g_euler_all, g_s])
    cosmo_euler_all = jnp.concatenate([cosmo_euler_all, c_s])

theta_euler_all = cosmo_euler_all / SCALE_COSMO * THETA_STD + THETA_MEAN

# --- sample_with_log_prob (returns log_prior_z0 and delta_logp separately) ---
def sample_with_logq(key, batch_size=16, num_steps=128, num_hutchinson=10):
    nnx.update(model, states)
    keys = jax.random.split(key, batch_size)
    def sample_one(k):
        return model.sample_with_log_prob(
            x_shape=(N, N, nbins), cosmo_shape=(6,),
            cond=cond, key=k, num_steps=num_steps, num_hutchinson=num_hutchinson,
        )
    g_norm, cosmo_norm, log_prior_z0, delta_logp = jax.vmap(sample_one)(keys)
    g_phys = g_norm * GRF_STD + GRF_MEAN
    theta_phys = cosmo_norm / SCALE_COSMO * THETA_STD + THETA_MEAN
    return g_phys, theta_phys, log_prior_z0, delta_logp

key = jax.random.key(0)
key, subkey = jax.random.split(key)
g_lp_all, theta_lp_all, log_prior_z0_all, delta_logp_all = sample_with_logq(subkey)
for i in tqdm(range(5), desc="LP batches"):
    key, subkey = jax.random.split(key)
    g_s, t_s, lp_s, dl_s = sample_with_logq(subkey)
    g_lp_all = jnp.concatenate([g_lp_all, g_s])
    theta_lp_all = jnp.concatenate([theta_lp_all, t_s])
    log_prior_z0_all = jnp.concatenate([log_prior_z0_all, lp_s])
    delta_logp_all = jnp.concatenate([delta_logp_all, dl_s])

print(f"\nEuler theta mean:  {theta_euler_all.mean(0)}")
print(f"LP theta mean:     {theta_lp_all.mean(0)}")

# --- Contour comparison ---
samples_mcmc_gd = MCSamples(samples=np.array(mcmc_samples), names=names, label="MCMC")
samples_euler_gd = MCSamples(samples=np.array(theta_euler_all), names=names, label="Euler")
samples_lp_gd = MCSamples(samples=np.array(theta_lp_all), names=names, label="Flow")

g_plot = plots.get_subplot_plotter()
g_plot.triangle_plot(
    [samples_euler_gd, samples_lp_gd, samples_mcmc_gd],
    names, markers=data['theta'], marker_args={"lw": 1},
    filled=[True, False, False],
    contour_colors=["#d06e99ff", "blue", "black"],
    contour_ls=["-", "-", "--"],
    contour_lws=[2., 2., 2.],
)
plt.savefig(os.path.join(save_dir, "contour_compare_samplers.png"))
print("Sampler comparison contours saved")

# ============================================================
# Compute target log prob components
# ============================================================
print("\n" + "=" * 60)
print("Computing target log prob")
print("=" * 60)

def compute_target_batched(g_samples, theta_samples, batch_size=4):
    lp_list, lpz_list, lzp_list, ld_list = [], [], [], []
    for i in tqdm(range(0, len(g_samples), batch_size), desc="target"):
        g_batch = jnp.transpose(g_samples[i:i+batch_size], [0, 3, 1, 2])
        lp, lpz, lzp, ld = jax.vmap(compute_log_p_g_and_components)(
            g_batch, theta_samples[i:i+batch_size])
        lp_list.append(lp)
        lpz_list.append(lpz)
        lzp_list.append(lzp)
        ld_list.append(ld)
    return (jnp.concatenate(lp_list), jnp.concatenate(lpz_list),
            jnp.concatenate(lzp_list), jnp.concatenate(ld_list))

log_p_all, log_pz_joint_all, log_pz_prior_all, log_det_L_all = compute_target_batched(
    g_lp_all, theta_lp_all)

# ============================================================
# Decomposition diagnostics
# ============================================================
print("\n--- Decomposition ---")
log_theta_lik_all = log_pz_joint_all - log_pz_prior_all

print(f"log p(theta,z,y) std:        {log_pz_joint_all.std():.1f}")
print(f"log p(z) std:                {log_pz_prior_all.std():.1f}")
print(f"log p(theta)+log p(y|x) std: {log_theta_lik_all.std():.1f}")
print(f"log|det dg/dz| std:          {log_det_L_all.std():.1f}")
print(f"log_prior_z0 std:            {log_prior_z0_all.std():.1f}")
print(f"delta_logp std:              {delta_logp_all.std():.1f}")

# ============================================================
# Full IS (for comparison)
# ============================================================
print("\n" + "=" * 60)
print("Full IS (g-space)")
print("=" * 60)

log_q_full = log_prior_z0_all + delta_logp_all + log_det_rescale_g + log_det_rescale_theta

valid = jnp.isfinite(log_p_all) & jnp.isfinite(log_q_full)
print(f"Valid: {valid.sum()} / {len(valid)}")

log_weights_full = jnp.where(valid, log_p_all - log_q_full, -jnp.inf)
log_w_full_norm = log_weights_full - jax.scipy.special.logsumexp(log_weights_full)
weights_full = jnp.exp(log_w_full_norm)
ess_full = 1.0 / jnp.sum(weights_full ** 2)
corr_full = jnp.corrcoef(log_p_all[valid], log_q_full[valid])[0, 1]

print(f"log_p std:       {log_p_all[valid].std():.1f}")
print(f"log_q std:       {log_q_full[valid].std():.1f}")
print(f"log_weights std: {log_weights_full[valid].std():.1f}")
print(f"corr(log_p, log_q): {corr_full:.4f}")
print(f"ESS (full): {ess_full:.1f} / {valid.sum()} ({100*ess_full/valid.sum():.1f}%)")

# ============================================================
# Prior-cancelled IS
# ============================================================
#
# Full:  log w = [log p(z) + log p(theta) + log p(y|x) - log|det_L|]
#              - [log prior(z0) + delta_logp + rescaling]
#
# Prior-cancelled (remove log p(z) and log prior(z0)):
#   log w = [log p(theta) + log p(y|x) - log|det_L|]
#         - [delta_logp + rescaling]
#
# log|det_L| varies only with theta (small std), not per sample.
# log p(theta)+log p(y|x) has std ~71 from x-space analysis.
#
print("\n" + "=" * 60)
print("Prior-cancelled IS (g-space)")
print("=" * 60)

# Target signal: log p(theta) + log p(y|x) - log|det_L|
# = log p(theta,z,y) - log p(z) - log|det_L|
# = log_p_all - log_pz_prior_all   (since log_p_all = log_pz_joint - log_det_L)
# Wait: log_p_all = log_pz_joint - log_det_L
# So:   log_p_all - log_pz_prior_all = log_pz_joint - log_det_L - log_pz_prior_all
#      = [log p(theta) + log p(y|x)] - log_det_L
# That still has log_det_L in it. But log_det_L only varies with theta (small std).
# 
# Actually more cleanly:
# log_p_signal = log p(theta) + log p(y|x) = log_pz_joint - log_pz_prior
log_p_signal = log_pz_joint_all - log_pz_prior_all

# Proposal signal: delta_logp + rescaling
log_q_signal = delta_logp_all + log_det_rescale_g + log_det_rescale_theta

# But we also need to account for log|det_L| which is in log_p_all but not log_p_signal
# Full:          log w = (log_pz_joint - log_det_L) - (log_prior_z0 + delta_logp + rescaling)
# Prior-cancel:  log w = (log_pz_joint - log_pz_prior - log_det_L) - (delta_logp + rescaling)
#                      = log_p_signal - log_det_L - log_q_signal
log_weights_pc = log_p_signal - log_det_L_all - log_q_signal

valid_pc = jnp.isfinite(log_weights_pc)
print(f"Valid: {valid_pc.sum()} / {len(valid_pc)}")

print(f"\nlog_p_signal std:             {log_p_signal[valid_pc].std():.1f}")
print(f"log_det_L std:                {log_det_L_all[valid_pc].std():.1f}")
print(f"log_p_signal - log_det_L std: {(log_p_signal - log_det_L_all)[valid_pc].std():.1f}")
print(f"log_q_signal std:             {log_q_signal[valid_pc].std():.1f}")
print(f"delta_logp std:               {delta_logp_all[valid_pc].std():.1f}")
print(f"log_prior_z0 std:             {log_prior_z0_all[valid_pc].std():.1f}")
print(f"log_pz_prior std:             {log_pz_prior_all[valid_pc].std():.1f}")
print(f"log_weights_pc std:           {log_weights_pc[valid_pc].std():.1f}")

corr_pc = jnp.corrcoef(
    (log_p_signal - log_det_L_all)[valid_pc], log_q_signal[valid_pc])[0, 1]
print(f"corr(log_p_signal - log_det_L, log_q_signal): {corr_pc:.4f}")

log_w_pc_valid = jnp.where(valid_pc, log_weights_pc, -jnp.inf)
log_w_pc_norm = log_w_pc_valid - jax.scipy.special.logsumexp(log_w_pc_valid)
weights_pc = jnp.exp(log_w_pc_norm)
ess_pc = 1.0 / jnp.sum(weights_pc ** 2)
print(f"ESS (prior-cancelled): {ess_pc:.1f} / {valid_pc.sum()} ({100*ess_pc/valid_pc.sum():.1f}%)")

# ============================================================
# Summary table
# ============================================================
print(f"\n{'':>35s} {'full':>16s} {'prior-cancelled':>16s}")
print(f"{'log_p std':>35s} {log_p_all[valid].std():>16.1f} {(log_p_signal - log_det_L_all)[valid_pc].std():>16.1f}")
print(f"{'log_q std':>35s} {log_q_full[valid].std():>16.1f} {log_q_signal[valid_pc].std():>16.1f}")
print(f"{'log_weights std':>35s} {log_weights_full[valid].std():>16.1f} {log_weights_pc[valid_pc].std():>16.1f}")
print(f"{'corr(log_p, log_q)':>35s} {corr_full:>16.4f} {corr_pc:>16.4f}")
print(f"{'ESS':>35s} {ess_full:>16.1f} {ess_pc:>16.1f}")

# ============================================================
# Scatter plots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(log_q_full[valid], log_p_all[valid], s=5, alpha=0.5)
axes[0].set_xlabel('log q (full)')
axes[0].set_ylabel('log p (full)')
axes[0].set_title(f'Full IS: corr={corr_full:.4f}, ESS={ess_full:.1f}')

axes[1].scatter(log_q_signal[valid_pc], (log_p_signal - log_det_L_all)[valid_pc], s=5, alpha=0.5)
axes[1].set_xlabel('log q (prior-cancelled)')
axes[1].set_ylabel('log p (prior-cancelled)')
axes[1].set_title(f'Prior-cancelled: corr={corr_pc:.4f}, ESS={ess_pc:.1f}')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'logp_vs_logq_gspace.png'))
print("Scatter plots saved")

# ============================================================
# Weighted posterior (use best weights)
# ============================================================
# Use prior-cancelled if ESS is better, otherwise full
if ess_pc > ess_full:
    weights_best = weights_pc
    valid_best = valid_pc
    label_best = "IS (prior-cancelled)"
    print(f"\nUsing prior-cancelled weights (ESS={ess_pc:.1f})")
else:
    weights_best = weights_full
    valid_best = valid
    label_best = "IS (full)"
    print(f"\nUsing full weights (ESS={ess_full:.1f})")

theta_mean = jnp.sum(weights_best[:, None] * theta_lp_all, axis=0)
theta_std = jnp.sqrt(jnp.sum(weights_best[:, None] * (theta_lp_all - theta_mean) ** 2, axis=0))
print(f"\nWeighted posterior ({label_best}):")
for name, m, s in zip(params_name, theta_mean, theta_std):
    print(f"  {name}: {m:.4f} ± {s:.4f}")

# ============================================================
# Final contour plot
# ============================================================
samples_is_best = MCSamples(
    samples=np.array(theta_lp_all[valid_best]),
    weights=np.array(weights_best[valid_best]),
    names=names, label=label_best,
)

g_plot2 = plots.get_subplot_plotter()
g_plot2.settings.axes_fontsize = 26
g_plot2.settings.axes_labelsize = 28
g_plot2.settings.legend_fontsize = 22
g_plot2.triangle_plot(
    [samples_lp_gd, samples_mcmc_gd, samples_is_best],
    names, markers=data['theta'], marker_args={"lw": 1},
    filled=[True, False, False],
    line_args=[
        {"ls": "-", "color": "#d06e99ff"},
        {"ls": "--", "color": "black"},
        {"ls": "-", "color": "blue"},
    ],
    contour_colors=["#d06e99ff", "black", "blue"],
    contour_ls=["-", "--", "-"],
    contour_lws=[2., 2., 2.],
)
plt.savefig(os.path.join(save_dir, "contour_plot_gspace.png"))
plt.savefig(os.path.join(save_dir, "contour_plot_gspace.pdf"))
print("Final g-space IS contours saved")

# ============================================================
# Additional diagnostics: correlation structure
# ============================================================
print("\n" + "=" * 60)
print("Correlation diagnostics")
print("=" * 60)

corr_priors = jnp.corrcoef(log_pz_prior_all, log_prior_z0_all)[0, 1]
print(f"corr(log p(z), log prior(z0)):   {corr_priors:.4f}")
print(f"log p(z) std:                    {log_pz_prior_all.std():.1f}")
print(f"log prior(z0) std:               {log_prior_z0_all.std():.1f}")
print(f"(log p(z) - log prior(z0)) std:  {(log_pz_prior_all - log_prior_z0_all).std():.1f}")

corr_det_delta = jnp.corrcoef(log_det_L_all, delta_logp_all)[0, 1]
print(f"\ncorr(log|det_L|, delta_logp):    {corr_det_delta:.4f}")
print(f"log|det_L| std:                  {log_det_L_all.std():.1f}")
print(f"delta_logp std:                  {delta_logp_all.std():.1f}")
print(f"(log|det_L| - delta_logp) std:   {(log_det_L_all - delta_logp_all).std():.1f}")

# What if we drop log p(z) and log prior(z0), but keep log|det_L| on both sides?
# log w = [log p(theta) + log p(y|x)] - [delta_logp + rescaling + log|det_L| - log p(z) + log prior(z0)]
# Since we're just rearranging, this is the same weight. But what about dropping only priors?
log_w_drop_priors = (log_pz_joint_all - log_pz_prior_all - log_det_L_all) - (delta_logp_all + log_det_rescale_g + log_det_rescale_theta)
print(f"\nDrop only priors (keep log|det_L| in target):")
print(f"  log_weights std: {log_w_drop_priors[valid].std():.1f}")

# What about dropping log|det_L| too? i.e. just compare signal vs delta_logp
log_w_drop_all = (log_pz_joint_all - log_pz_prior_all) - (delta_logp_all + log_det_L_all + log_det_rescale_g + log_det_rescale_theta)
print(f"\nDrop priors AND move log|det_L| to q side:")
print(f"  log_weights std: {log_w_drop_all[valid].std():.1f}")
# Note: this is NOT the same as the original weight — we've dropped log p(z) - log prior(z0)

# The truly reduced weight: drop log p(z), drop log prior(z0), drop log|det_L| from both
# log w_reduced = log p(theta) + log p(y|x) - delta_logp - rescaling
log_w_reduced = log_p_signal - (delta_logp_all + log_det_rescale_g + log_det_rescale_theta)
corr_reduced = jnp.corrcoef(log_p_signal[valid], (delta_logp_all + log_det_rescale_g + log_det_rescale_theta)[valid])[0, 1]
log_w_reduced_valid = jnp.where(valid, log_w_reduced, -jnp.inf)
log_w_reduced_norm = log_w_reduced_valid - jax.scipy.special.logsumexp(log_w_reduced_valid)
weights_reduced = jnp.exp(log_w_reduced_norm)
ess_reduced = 1.0 / jnp.sum(weights_reduced ** 2)

print(f"\nReduced weight (drop all known terms):")
print(f"  log_p_signal std:   {log_p_signal[valid].std():.1f}")
print(f"  delta_logp std:     {delta_logp_all[valid].std():.1f}")
print(f"  log_weights std:    {log_w_reduced[valid].std():.1f}")
print(f"  corr:               {corr_reduced:.4f}")
print(f"  ESS:                {ess_reduced:.1f} / {valid.sum()} ({100*ess_reduced/valid.sum():.1f}%)")