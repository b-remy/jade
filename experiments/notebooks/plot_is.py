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
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.flow import Denoiser
from jade.utils import load_model
from jade.sampling import EulerSampler

import jax_cosmo as jc
from numpyro.handlers import condition, seed, trace
from numpyro.infer.util import log_density
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from jade.log_prob import log_prob_x, compute_lensing_jacobian

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
# Load model
# ============================================================
cfg, states = load_model(
    "/u/bremy/repos/jade/experiments/wandb/run-20260219_232046-jhj5rm2p/files/checkpoints",
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

model_log_normal_sim = partial(
    lensingLogNormal,
    N=N, map_size=map_size, gal_per_arcmin2=gals_per_arcmin2,
    sigma_e=sigma_e, nbins=nbins, a=a, b=b_nz, z0=z0,
    model_type='lognormal', lognormal_shifts='LSSTY10', with_noise=True,
)

cond_model_sim = seed(model_log_normal_sim, key)
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
# Forward model and target log prob
# ============================================================
model_log_normal = partial(
    lensingLogNormal,
    N=N, map_size=map_size, gal_per_arcmin2=gals_per_arcmin2,
    sigma_e=sigma_e, nbins=nbins, a=a, b=b_nz, z0=z0,
    model_type='lognormal', lognormal_shifts='LSSTY10', with_noise=True,
)

fwd_kwargs = dict(
    N=N, nbins=nbins, map_size=map_size,
    gal_per_arcmin2=gals_per_arcmin2, sigma_e=sigma_e,
    a=a, b=b_nz, z0=z0, lognormal_shifts='LSSTY10',
)

conditioned_model = condition(model_log_normal, data={"y": obs})

def log_prob_z_fn(params):
    return log_density(conditioned_model, (), {}, params)[0]

# Rescaling constants (flow normalized space <-> physical space)
log_det_rescale_x = -N * N * jnp.sum(jnp.log(FIELD_STD))
log_det_rescale_theta = -jnp.sum(jnp.log(THETA_STD / SCALE_COSMO))

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
    x_0 = jax.random.normal(keys[0], shape=(batch_size, N, N, nbins))
    cosmo_0 = jax.random.normal(keys[1], shape=(batch_size, 6))
    sample_keys = jax.random.split(keys[2], batch_size)
    x_s, c_s = jax.vmap(euler, in_axes=(0, 0, None, 0))(x_0, cosmo_0, cond, sample_keys)
    return x_s, c_s

key = jax.random.key(0)
x_euler_all, cosmo_euler_all = sample_euler(key)
for i in tqdm(range(3), desc="Euler batches"):
    key, subkey = jax.random.split(key)
    x_s, c_s = sample_euler(subkey)
    x_euler_all = jnp.concatenate([x_euler_all, x_s])
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
    x_norm, cosmo_norm, log_prior_z0, delta_logp = jax.vmap(sample_one)(keys)
    x_phys = x_norm * FIELD_STD + FIELD_MEAN
    theta_phys = cosmo_norm / SCALE_COSMO * THETA_STD + THETA_MEAN
    return x_phys, theta_phys, log_prior_z0, delta_logp

key = jax.random.key(0)
key, subkey = jax.random.split(key)
x_lp_all, theta_lp_all, log_prior_z0_all, delta_logp_all = sample_with_logq(subkey)
for i in tqdm(range(5), desc="LP batches"):
    key, subkey = jax.random.split(key)
    x_s, t_s, lp_s, dl_s = sample_with_logq(subkey)
    x_lp_all = jnp.concatenate([x_lp_all, x_s])
    theta_lp_all = jnp.concatenate([theta_lp_all, t_s])
    log_prior_z0_all = jnp.concatenate([log_prior_z0_all, lp_s])
    delta_logp_all = jnp.concatenate([delta_logp_all, dl_s])

print(f"\nEuler theta mean:  {theta_euler_all.mean(0)}")
print(f"LP theta mean:     {theta_lp_all.mean(0)}")

# --- Contour comparison ---
samples_mcmc_gd = MCSamples(samples=np.array(mcmc_samples), names=names, label="MCMC")
samples_euler_gd = MCSamples(samples=np.array(theta_euler_all), names=names, label="Euler")
samples_lp_gd = MCSamples(samples=np.array(theta_lp_all), names=names, label="Flow")

g = plots.get_subplot_plotter()
g.triangle_plot(
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
# Importance sampling (prior-cancelled formulation)
# ============================================================
#
# Full IS:
#   log w = log p(theta,z,y) - log|det dx/dz| - log q(theta,x|y)
#         = [log p(z) + log p(theta) + log p(y|x)] - log|det| - [log prior(z0) + delta_logp + rescaling]
#
# log p(z) std ~1657, log prior(z0) std ~1657: both are -0.5*||.||^2 over 82k dims
# They add ~1657 of pure noise to weights, drowning the signal (std ~71).
#
# Prior-cancelled IS (remove both Gaussian prior terms):
#   log w = [log p(theta) + log p(y|x)] - [delta_logp + rescaling]
#
print("\n" + "=" * 60)
print("Importance sampling (prior-cancelled)")
print("=" * 60)

# --- Compute target terms ---
@jax.jit
def compute_all_target_terms(x, theta):
    L, shift, z, log_det = compute_lensing_jacobian(theta, x=x, **fwd_kwargs)
    omega_c, omega_b, sigma_8, h_0, n_s, w_0 = theta
    params = {"omega_c": omega_c, "omega_b": omega_b, "sigma_8": sigma_8,
              "h_0": h_0, "n_s": n_s, "w_0": w_0, "z": z}
    log_pz = log_prob_z_fn(params)
    log_p_z_prior = -0.5 * jnp.sum(z**2) - 0.5 * z.size * jnp.log(2*jnp.pi)
    return log_pz, log_p_z_prior, log_det

def compute_target_batched(x_samples, theta_samples, batch_size=4):
    lpz_list, lzp_list, ld_list = [], [], []
    for i in tqdm(range(0, len(x_samples), batch_size), desc="target"):
        lpz, lzp, ld = jax.vmap(compute_all_target_terms)(
            x_samples[i:i+batch_size], theta_samples[i:i+batch_size])
        lpz_list.append(lpz)
        lzp_list.append(lzp)
        ld_list.append(ld)
    return jnp.concatenate(lpz_list), jnp.concatenate(lzp_list), jnp.concatenate(ld_list)

log_pz_all, log_pz_prior_all, log_det_all = compute_target_batched(x_lp_all, theta_lp_all)

# --- Signal part of target: log p(theta) + log p(y|x) ---
log_p_signal = log_pz_all - log_pz_prior_all

# --- Signal part of proposal: delta_logp + rescaling ---
log_q_signal = delta_logp_all + log_det_rescale_x + log_det_rescale_theta

# --- Prior-cancelled IS weights ---
log_weights_pc = log_p_signal - log_q_signal

valid = jnp.isfinite(log_weights_pc)
print(f"Valid: {valid.sum()} / {len(valid)}")

# --- Full IS weights (for comparison) ---
log_q_full = log_prior_z0_all + delta_logp_all + log_det_rescale_x + log_det_rescale_theta
log_p_full = log_pz_all - log_det_all
log_weights_full = jnp.where(valid, log_p_full - log_q_full, -jnp.inf)

# --- Diagnostics ---
print(f"\nlog_p_signal std:     {log_p_signal[valid].std():.1f}")
print(f"log_q_signal std:     {log_q_signal[valid].std():.1f}")
print(f"delta_logp std:       {delta_logp_all[valid].std():.1f}")
print(f"log_prior_z0 std:     {log_prior_z0_all[valid].std():.1f}")
print(f"log_pz_prior std:     {log_pz_prior_all[valid].std():.1f}")
print(f"log_weights_pc std:   {log_weights_pc[valid].std():.1f}")

corr_pc = jnp.corrcoef(log_p_signal[valid], log_q_signal[valid])[0, 1]
print(f"corr(log_p_signal, log_q_signal): {corr_pc:.4f}")

# ESS for prior-cancelled
log_w_pc_valid = jnp.where(valid, log_weights_pc, -jnp.inf)
log_w_pc_norm = log_w_pc_valid - jax.scipy.special.logsumexp(log_w_pc_valid)
weights_pc = jnp.exp(log_w_pc_norm)
ess_pc = 1.0 / jnp.sum(weights_pc ** 2)

# ESS for full IS
log_w_full_valid = jnp.where(valid, log_weights_full, -jnp.inf)
log_w_full_norm = log_w_full_valid - jax.scipy.special.logsumexp(log_w_full_valid)
weights_full = jnp.exp(log_w_full_norm)
ess_full = 1.0 / jnp.sum(weights_full ** 2)
corr_full = jnp.corrcoef(log_p_full[valid], log_q_full[valid])[0, 1]

# Summary table
print(f"\n{'':>30s} {'prior-cancelled':>16s} {'full':>16s}")
print(f"{'log_p std':>30s} {log_p_signal[valid].std():>16.1f} {log_p_full[valid].std():>16.1f}")
print(f"{'log_q std':>30s} {log_q_signal[valid].std():>16.1f} {log_q_full[valid].std():>16.1f}")
print(f"{'log_weights std':>30s} {log_weights_pc[valid].std():>16.1f} {log_weights_full[valid].std():>16.1f}")
print(f"{'corr(log_p, log_q)':>30s} {corr_pc:>16.4f} {corr_full:>16.4f}")
print(f"{'ESS':>30s} {ess_pc:>16.1f} {ess_full:>16.1f}")

# --- Scatter plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(log_q_full[valid], log_p_full[valid], s=5, alpha=0.5)
axes[0].set_xlabel('log q (full)')
axes[0].set_ylabel('log p (full)')
axes[0].set_title(f'Full IS: corr={corr_full:.4f}, ESS={ess_full:.1f}')

axes[1].scatter(log_q_signal[valid], log_p_signal[valid], s=5, alpha=0.5)
axes[1].set_xlabel('log q (prior-cancelled)')
axes[1].set_ylabel('log p (prior-cancelled)')
axes[1].set_title(f'Prior-cancelled: corr={corr_pc:.4f}, ESS={ess_pc:.1f}')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'logp_vs_logq_comparison.png'))
print("Scatter plots saved")

# ============================================================
# Weighted posterior (prior-cancelled weights)
# ============================================================
theta_mean_pc = jnp.sum(weights_pc[:, None] * theta_lp_all, axis=0)
theta_std_pc = jnp.sqrt(jnp.sum(weights_pc[:, None] * (theta_lp_all - theta_mean_pc) ** 2, axis=0))
print(f"\nWeighted posterior (prior-cancelled IS):")
for name, m, s in zip(params_name, theta_mean_pc, theta_std_pc):
    print(f"  {name}: {m:.4f} ± {s:.4f}")

# ============================================================
# Final contour plots
# ============================================================
samples_is_pc = MCSamples(
    samples=np.array(theta_lp_all[valid]),
    weights=np.array(weights_pc[valid]),
    names=names, label="IS (prior-cancelled)",
)

g = plots.get_subplot_plotter()
g.settings.axes_fontsize = 26
g.settings.axes_labelsize = 28
g.settings.legend_fontsize = 22
g.triangle_plot(
    [samples_lp_gd, samples_mcmc_gd, samples_is_pc],
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
plt.savefig(os.path.join(save_dir, "contour_plot_prior_cancelled.png"))
plt.savefig(os.path.join(save_dir, "contour_plot_prior_cancelled.pdf"))
print("Final prior-cancelled IS contours saved")