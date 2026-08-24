"""Per-t denoiser MSE: diagnose under-trained noise regimes.

For a checkpoint, sweeps diffusion time t in [0.05, 0.95] and computes the
per-sample MSE of (x̂, θ̂) vs (x_0, θ_0). Plots:
  panel 1 — training time density p(t) implied by cfg.diffusion.{mu,sigma}
  panel 2 — κ denoiser MSE vs t (mean ± std over noise realizations)
  panel 3 — θ denoiser MSE vs t (overall + per-cosmo-param)

The hypothesis: if joint training starves the low-noise (high-t) regime,
θ MSE will rise / plateau where training mass is sparse — confirming that
the time distribution is the cause of unstable cosmology marginals.

Usage:
  python diagnose_t_grid.py --checkpoint <ckpt_dir> [--params-name X] \\
                            [--dataset PATH] [--n-samples 128] \\
                            [--n-t 19] [--n-noise 4] [--out FILE.pdf]
"""

import os
import argparse

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import matplotlib.pyplot as plt

from jade.nn_hybrid import JADE_B_16
from jade.flow import Denoiser
from jade.utils import load_model
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, sigma_lsst
from datasets import load_from_disk


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
        "Per-t denoiser MSE — diagnose under-trained noise regimes.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir")
    parser.add_argument("--params-name", default="JADE_B_16_ema_latest")
    parser.add_argument("--dataset", default="../sbi_lens_full")
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--n-t", type=int, default=19,
                        help="Number of t values from 0.05 to 0.95")
    parser.add_argument("--n-noise", type=int, default=4,
                        help="Independent noise realizations per t")
    parser.add_argument("--out", default="t_grid_mse.pdf")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint} / {args.params_name}")
    cfg, states = load_model(args.checkpoint, args.params_name)
    model = make_model(cfg, nnx.Rngs(cfg['training']['seed']))
    model = Denoiser(model, cfg)
    nnx.update(model, states)

    dataset = load_from_disk(args.dataset).with_format("numpy")
    batch = next(dataset.iter(batch_size=args.n_samples))

    x = (batch["map"] - FIELD_MEAN.reshape(1, 1, 1, -1)) / FIELD_STD.reshape(1, 1, 1, -1)
    cosmo = (batch["theta"] - THETA_MEAN) / THETA_STD

    key = jax.random.key(0)
    key, sk = jax.random.split(key)
    map_phys = batch["map"]
    noise = sigma_lsst.reshape(1, 1, 1, -1) * jax.random.normal(sk, shape=map_phys.shape)
    cond = ((map_phys + noise) - FIELD_MEAN.reshape(1, 1, 1, -1)) / FIELD_STD.reshape(1, 1, 1, -1)

    t_grid = jnp.linspace(0.05, 0.95, args.n_t)
    n = args.n_samples

    @nnx.jit
    def eval_step(model, x, cosmo, cond, t_vec, ks):
        xt, cosmot = jax.vmap(model.forward_coupling, in_axes=(0, 0, 0, 0))(
            x, cosmo, t_vec, ks)
        x_pred, cosmo_pred = jax.vmap(
            model.x_pred, in_axes=(0, 0, 0, 0, None))(xt, cosmot, t_vec, cond, False)
        mse_x = jnp.mean((x_pred - x) ** 2)
        mse_cosmo_per_dim = jnp.mean((cosmo_pred - cosmo) ** 2, axis=0)
        return mse_x, mse_cosmo_per_dim

    mse_x_all = np.zeros((args.n_t, args.n_noise))
    mse_cosmo_pd_all = np.zeros((args.n_t, args.n_noise, 6))

    print(f"Sweeping {args.n_t} t-values × {args.n_noise} noise realizations "
          f"on {n} samples...")
    for ti, t_val in enumerate(t_grid):
        t_vec = jnp.full((n,), t_val)
        for ni in range(args.n_noise):
            key, sk = jax.random.split(key)
            ks = jax.random.split(sk, n)
            mse_x, mse_cosmo_pd = eval_step(model, x, cosmo, cond, t_vec, ks)
            mse_x_all[ti, ni] = float(mse_x)
            mse_cosmo_pd_all[ti, ni] = np.asarray(mse_cosmo_pd)
        print(f"  t={float(t_val):.3f}  σ={1-float(t_val):.3f}  "
              f"MSE_κ={mse_x_all[ti].mean():.4e}  "
              f"MSE_θ={mse_cosmo_pd_all[ti].mean():.4e}")

    # --- training time density implied by config ---
    # FlowLoss draws s = (N(0,1) + mu) * sigma  →  s ~ N(sigma*mu, sigma²)
    # then t = sigmoid(s). Push-forward density: p(t) = p(s) / (t*(1-t)).
    mu = float(cfg['diffusion']['mu'])
    sig = float(cfg['diffusion']['sigma'])
    t_dense = np.linspace(0.001, 0.999, 800)
    s_at_t = np.log(t_dense / (1 - t_dense))
    p_s = (1.0 / np.sqrt(2 * np.pi * sig ** 2)) * np.exp(
        -(s_at_t - sig * mu) ** 2 / (2 * sig ** 2)
    )
    p_t = p_s / (t_dense * (1 - t_dense))

    t_np = np.asarray(t_grid)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # panel 1: training density
    axes[0].plot(t_dense, p_t, color='tab:purple', linewidth=2)
    axes[0].fill_between(t_dense, 0, p_t, alpha=0.3, color='tab:purple')
    axes[0].set_xlabel("t  (clean=1, noise=0)")
    axes[0].set_ylabel("training density  p(t)")
    axes[0].set_title(f"Training time distribution (μ={mu}, σ={sig})")
    axes[0].set_xlim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # panel 2: κ MSE
    mean_x = mse_x_all.mean(axis=1)
    std_x = mse_x_all.std(axis=1)
    axes[1].plot(t_np, mean_x, '-o', color='tab:blue')
    axes[1].fill_between(t_np, mean_x - std_x, mean_x + std_x,
                         alpha=0.3, color='tab:blue')
    axes[1].set_xlabel("t  (clean=1, noise=0)")
    axes[1].set_ylabel(r"MSE  $\|\hat\kappa - \kappa_0\|^2$")
    axes[1].set_title("κ denoiser error")
    axes[1].set_yscale('log')
    axes[1].set_xlim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # panel 3: θ MSE — overall + per-param
    overall_mean = mse_cosmo_pd_all.mean(axis=(1, 2))
    overall_std = mse_cosmo_pd_all.mean(axis=2).std(axis=1)
    axes[2].plot(t_np, overall_mean, '-o', color='black', linewidth=2.5,
                 label='avg of 6', zorder=10)
    axes[2].fill_between(t_np, overall_mean - overall_std,
                         overall_mean + overall_std,
                         alpha=0.3, color='black', zorder=9)
    param_names = [r'$\Omega_c$', r'$\Omega_b$', r'$\sigma_8$',
                   r'$h_0$', r'$n_s$', r'$w_0$']
    per_param_mean = mse_cosmo_pd_all.mean(axis=1)
    for i, name in enumerate(param_names):
        axes[2].plot(t_np, per_param_mean[:, i], '--', alpha=0.7, label=name)
    axes[2].set_xlabel("t  (clean=1, noise=0)")
    axes[2].set_ylabel(r"MSE  $\|\hat\theta - \theta_0\|^2$  (normalized θ)")
    axes[2].set_title("θ denoiser error per parameter")
    axes[2].set_yscale('log')
    axes[2].set_xlim(0, 1)
    axes[2].legend(fontsize=8, loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Per-t denoiser diagnostic — "
        f"{os.path.basename(args.checkpoint.rstrip('/'))} / {args.params_name}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches='tight')
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
