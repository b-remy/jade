"""Posterior calibration diagnostic.

For a fixed observation (the same MCMC reference plot_amortized.py uses),
samples the JADE posterior at one or more sampler endpoints t_max, and
compares per-parameter against the reference MCMC chain.

Two views:
  (A) Per-param table at every t_max — JADE_std / MCMC_std, bias vs MCMC mean,
      bias vs truth (in units of MCMC σ).
  (B) Figure: posterior std vs t_max per parameter, with MCMC reference line.

Reading the figure:
  - JADE std DROPS then RISES across t_max → late-trajectory steps over-spread
    (high-t score field issue) — fix: better high-t denoising.
  - JADE std monotonically DROPS but plateaus ABOVE MCMC → mid-trajectory
    error accumulation — fix: loss reweighting or capacity.
  - Per-param: only some parameters underconfident → parameter-specific
    issue (lambda_cosmo, info content, etc.).

Usage:
  python diagnose_posterior.py --checkpoint <ckpt_dir> [--params-name X] \\
      [--mcmc-dir PATH] [--n-samples 256] [--num-steps 50] \\
      [--t-maxes "0.7,0.85,0.9,0.95,0.98,1.0"] [--out FILE.pdf]
"""

import os
import argparse
import pickle

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import matplotlib.pyplot as plt

from jade.nn_hybrid import JADE_B_16
from jade.flow import Denoiser
from jade.utils import load_model
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.sampling import HeunSampler


def make_model(cfg, rngs):
    return JADE_B_16(
        rngs=rngs,
        in_channels=cfg['model']['in_channels'],
        input_size=cfg['model']['input_size'],
        enable_cond_image=cfg['model']['enable_cond_image'],
        cond_channels=cfg['model']['cond_channels'],
        num_cosmo_tokens=cfg['model']['num_cosmo_tokens'],
        cond_patch_size=cfg['model']['cond_patch_size'],
        cond_start=cfg['model']['cond_start'],
        attn_drop=cfg['model']['attn_drop'],
        proj_drop=cfg['model']['proj_drop'],
        split_qkv=cfg['model'].get('split_qkv', False),
        mask_theta_to_field=cfg['model'].get('mask_theta_to_field', False),
    )


def main():
    parser = argparse.ArgumentParser(description=
        "Per-parameter posterior calibration vs MCMC.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir")
    parser.add_argument("--params-name", default="JADE_B_16_ema_best")
    parser.add_argument("--mcmc-dir", default="../mcmc_log_normal")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--t-maxes", default="0.7,0.85,0.9,0.95,0.98,1.0",
                        help="Comma-separated list of t_max values to sweep")
    parser.add_argument("--out", default="posterior_diagnose.pdf")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint} / {args.params_name}")
    cfg, states = load_model(args.checkpoint, args.params_name)
    model = make_model(cfg, nnx.Rngs(cfg['training']['seed']))
    model = Denoiser(model, cfg)
    nnx.update(model, states)

    # MCMC reference
    with open(os.path.join(args.mcmc_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
        ref = pickle.load(f)
    with open(os.path.join(args.mcmc_dir, "mcmc_log_posterior_samples.pkl"), "rb") as f:
        mcmc_samples = pickle.load(f)

    obs = jnp.asarray(ref['y'])
    truth = np.asarray(ref['theta'])
    cond = (obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)

    mcmc_phys = np.asarray(mcmc_samples)
    mcmc_mean = mcmc_phys.mean(0)
    mcmc_std = mcmc_phys.std(0)

    t_maxes = [float(x) for x in args.t_maxes.split(',')]

    SCALE_COSMO = float(cfg['loss'].get('SCALE_COSMO', 1.0))

    results = {}  # t_max -> phys-units posterior samples (n_samples, 6)

    for t_max in t_maxes:
        sampler = HeunSampler(model=model, num_steps=args.num_steps, t_max=t_max)
        key = jax.random.key(0)
        keys = jax.random.split(key, 3)
        x_0 = jax.random.normal(keys[0], shape=(args.n_samples, 128, 128, 5))
        cosmo_0 = jax.random.normal(keys[1], shape=(args.n_samples, 6))
        sk_keys = jax.random.split(keys[2], args.n_samples)

        _, cosmo_samples = jax.vmap(sampler, in_axes=(0, 0, None, 0))(
            x_0, cosmo_0, cond, sk_keys)

        # Raw sampler output is α_t · θ_0 + σ_t · ε_residue (in scaled-norm θ).
        # We do NOT rescale by 1/α_t — that amplifies the residue and obscures
        # the comparison. Instead, report stats on the raw output and let the
        # bias term reveal the truncation effect.
        cosmo_norm = np.asarray(cosmo_samples) / SCALE_COSMO
        cosmo_phys = cosmo_norm * THETA_STD + THETA_MEAN
        results[t_max] = cosmo_phys

    param_names = [r'$\Omega_c$', r'$\Omega_b$', r'$\sigma_8$',
                   r'$h_0$', r'$n_s$', r'$w_0$']
    plain_names = ['Omega_c', 'Omega_b', 'sigma_8', 'h_0', 'n_s', 'w_0']

    # ---- table ----
    print()
    print("=" * 102)
    print("Per-parameter posterior calibration vs MCMC")
    print(f"  MCMC: n_samples = {len(mcmc_phys)}, mean = {mcmc_mean.round(4)}, "
          f"std = {mcmc_std.round(4)}")
    print(f"  Truth: {truth.round(4)}")
    print(f"  JADE: n_samples = {args.n_samples}, num_steps = {args.num_steps}")
    print("=" * 102)
    print(f"  {'param':<10s}{'JADE std':>12s}{'MCMC std':>12s}"
          f"{'std ratio':>12s}{'mean bias':>14s}{'bias/σ_MCMC':>14s}{'bias/truth':>14s}")
    print("-" * 102)
    for tm in t_maxes:
        print(f"\n  t_max = {tm:.2f}")
        for i, name in enumerate(plain_names):
            j = results[tm][:, i]
            j_mean, j_std = j.mean(), j.std()
            ratio = j_std / mcmc_std[i]
            bias_mcmc = (j_mean - mcmc_mean[i]) / mcmc_std[i]
            bias_truth = (j_mean - truth[i]) / mcmc_std[i]
            print(f"  {name:<10s}{j_std:>12.4f}{mcmc_std[i]:>12.4f}"
                  f"{ratio:>12.3f}{j_mean - mcmc_mean[i]:>14.4f}"
                  f"{bias_mcmc:>14.3f}{bias_truth:>14.3f}")
    print("=" * 102)
    print()

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, (name, ax) in enumerate(zip(param_names, axes.flat)):
        stds = [results[tm][:, i].std() for tm in t_maxes]
        ax.plot(t_maxes, stds, '-o', color='tab:blue', label='JADE')
        ax.axhline(mcmc_std[i], color='k', linestyle='--', label='MCMC')
        ax.set_title(name, fontsize=14)
        ax.set_xlabel(r'$t_{\max}$', fontsize=12)
        ax.set_ylabel(r'posterior std', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"Posterior std vs $t_{{\\max}}$ — "
        f"{os.path.basename(args.checkpoint.rstrip('/'))} / {args.params_name}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches='tight')
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
