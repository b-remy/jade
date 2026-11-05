#!/usr/bin/env python
"""Posterior sampling by likelihood guidance of the *unconditional* joint model.

`sample_posterior.py` uses the conditional network, which was trained to see the
observation. This script instead takes the unconditional model of p(theta, kappa)
and folds in an explicit Gaussian likelihood at sampling time, via
`jade.flow.PosteriorDenoiser` (MMPS or DPS). The forward operator is the Kaiser &
Squires convergence-to-shear map and the noise level is the LSST-Y10 per-bin
shape noise `jade.init.sigma_lsst`.

The observation is the same one the reference NUTS chain used, read from
`mcmc_log_normal/mcmc_log_obs_truth.pkl`, so the contours here, the conditional
model's contours, and the MCMC contours all describe the same posterior.

The `y` in that pickle is a *noisy convergence* map. The likelihood is written in
shear, so it is mapped through the KS inversion first. That transform is a
Fourier-space rotation, unitary on every mode except the mean, so the noise stays
white with the same per-bin std and `sigma_lsst` carries over unchanged.

Output is the same `.npz` schema as `sample_posterior.py`, so
`plot_paper_figures.py` reads it directly.

Examples
--------
    python sample_posterior_guidance.py --method mmps
    python sample_posterior_guidance.py --method dps --num-steps 512
"""

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from jade.flow import Denoiser, PosteriorDenoiser
from jade.init import FIELD_MEAN, FIELD_STD, THETA_MEAN, THETA_STD, sigma_lsst
from jade.lensing import ks93inv
from jade.nn import JADE_B_16
from jade.paths import FIGURES_DIR, MCMC_REF_DIR, WANDB_DIR
from jade.sampling import EulerSampler, HeunSampler
from jade.utils import load_model

N_PIX = 128
N_BINS = 5
COSMO_DIM = 6

# Last unconditional training run: configs/unconditional.yaml, 600 epochs,
# from scratch. This is the model that has no conditioning branch at all, so
# guidance is the only way to condition it on data.
UNCONDITIONAL_RUN = "run-20260904_113651-stwf6n94"


def build_model(ckpt, tag):
    """Rebuild the unconditional JADE and load `tag` into it."""
    cfg, states = load_model(ckpt, tag)

    if cfg["model"].get("enable_cond_image", False):
        raise ValueError(
            f"{ckpt} is a conditional checkpoint (enable_cond_image=true). "
            "Guidance is for the unconditional model; use sample_posterior.py instead."
        )

    model = JADE_B_16(
        rngs=nnx.Rngs(cfg["training"]["seed"]),
        in_channels=cfg["model"]["in_channels"],
        input_size=cfg["model"]["input_size"],
        patch_size=cfg["model"]["patch_size"],
        enable_cond_image=False,
        num_cosmo_tokens=cfg["model"]["num_cosmo_tokens"],
        attn_drop=cfg["model"]["attn_drop"],
        proj_drop=cfg["model"]["proj_drop"],
        attn_impl=cfg["model"].get("attn_impl", "dense"),
    )
    model = Denoiser(model, cfg)
    nnx.update(model, states)
    return cfg, model


def reference_field(key):
    """A noiseless convergence field at Planck15, for the figures' colour scale
    and their 2-point reference. Identical to sample_posterior.py."""
    from functools import partial

    import jax_cosmo as jc
    from numpyro.handlers import condition, seed, trace
    from sbi_lens.config import config_lsst_y_10
    from sbi_lens.simulator.LogNormal_field import lensingLogNormal

    cosmo = jc.parameters.Planck15()
    simulator = partial(
        lensingLogNormal,
        N=N_PIX,
        map_size=5,
        gal_per_arcmin2=config_lsst_y_10.gals_per_arcmin2,
        sigma_e=config_lsst_y_10.sigma_e,
        nbins=config_lsst_y_10.nbins,
        a=config_lsst_y_10.a,
        b=config_lsst_y_10.b,
        z0=config_lsst_y_10.z0,
        model_type="lognormal",
        lognormal_shifts="LSSTY10",
        with_noise=False,
    )
    model = condition(
        seed(simulator, key),
        {
            "omega_c": cosmo.Omega_c,
            "omega_b": cosmo.Omega_b,
            "sigma_8": cosmo.sigma8,
            "h_0": cosmo.h,
            "n_s": cosmo.n_s,
            "w_0": cosmo.w0,
        },
    )
    return np.asarray(trace(model).get_trace()["y"]["value"])


def shear_observation(obs):
    """Noisy convergence (N, N, n_bins) -> noisy shear (2, N, N, n_bins)."""
    g1, g2 = ks93inv(jnp.asarray(obs), jnp.zeros_like(jnp.asarray(obs)))
    return jnp.stack([g1, g2], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default=str(WANDB_DIR / UNCONDITIONAL_RUN / "files" / "checkpoints"),
        help="Checkpoint directory of an unconditional run.",
    )
    parser.add_argument("--ckpt-tag", default="JADE_B_16_ema_latest")
    parser.add_argument("--method", choices=["mmps", "dps"], default="mmps")
    parser.add_argument(
        "--sampler",
        choices=["euler", "heun"],
        default="euler",
        help="Heun costs two guided evaluations per step.",
    )
    # 512 is not a comfort margin: MMPS diverges outright at 128 Euler steps.
    # It is also not worth exceeding. DPS is converged in step count by 512 --
    # 1024 and 2048 reproduce it to three decimals, so its residual bias is the
    # method, not the integrator. See diagnose_guidance.py.
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--cg-maxiter", type=int, default=10)
    parser.add_argument(
        "--gmres-restart",
        type=int,
        default=20,
        help="GMRES inner Krylov dimension; sets the cost floor in denoiser VJPs per solve.",
    )
    parser.add_argument("--cg-tol", type=float, default=1e-3)
    parser.add_argument(
        "--solver",
        choices=["auto", "cg", "gmres"],
        default="auto",
        help="auto = GMRES for MMPS (non-symmetric operator; CG diverges on 20%% of draws), "
        "CG for DPS (operator is SPD, so CG is exact and cheap).",
    )
    parser.add_argument("--alpha-eps", type=float, default=1e-3, help="Floor on alpha_t in cov_t.")
    parser.add_argument(
        "--guide-cosmo",
        action="store_true",
        help="Also apply the guidance update to theta (the complete first-order update). "
        "Off by default, matching the exps-branch prototype.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcmc-dir", default=str(MCMC_REF_DIR))
    parser.add_argument("--out", default=None, help="Defaults to FIGURES_DIR/posterior_<method>.npz")
    args = parser.parse_args()

    out = args.out or str(FIGURES_DIR / f"posterior_{args.method}.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(jax.devices())

    cfg, model = build_model(args.ckpt, args.ckpt_tag)
    scale_cosmo = cfg["loss"]["SCALE_COSMO"]

    with open(os.path.join(args.mcmc_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
        ref = pickle.load(f)
    with open(os.path.join(args.mcmc_dir, "mcmc_log_posterior_samples.pkl"), "rb") as f:
        mcmc_samples = pickle.load(f)

    obs = ref["y"]
    gamma = shear_observation(obs)
    print(f"observation {np.shape(obs)} -> shear {gamma.shape}")
    print(f"sigma_gamma per bin: {np.array2string(np.asarray(sigma_lsst), precision=5)}")

    pdenoiser = PosteriorDenoiser(
        model=model,
        cfg=cfg,
        gamma=gamma,
        sigma_gamma=sigma_lsst,
        method=args.method,
        solver=args.solver,
        tol=args.cg_tol,
        maxiter=args.cg_maxiter,
        restart=args.gmres_restart,
        alpha_eps=args.alpha_eps,
        guide_cosmo=args.guide_cosmo,
    )
    sampler_cls = EulerSampler if args.sampler == "euler" else HeunSampler
    sampler = sampler_cls(model=pdenoiser, num_steps=args.num_steps)
    print(
        f"{args.method.upper()} guidance | {args.sampler} x {args.num_steps} steps | "
        f"{pdenoiser.solver} maxiter={args.cg_maxiter} tol={args.cg_tol} | "
        f"alpha_eps={args.alpha_eps} | guide_cosmo={args.guide_cosmo}"
    )

    @jax.jit
    def draw(key):
        keys = jax.random.split(key, 3)
        x_0 = jax.random.normal(keys[0], shape=(args.batch_size, N_PIX, N_PIX, N_BINS))
        cosmo_0 = jax.random.normal(keys[1], shape=(args.batch_size, COSMO_DIM))
        return jax.vmap(sampler, in_axes=(0, 0, None, 0))(
            x_0, cosmo_0, None, jax.random.split(keys[2], args.batch_size)
        )

    key = jax.random.key(args.seed)
    kappa, theta = [], []
    for i in tqdm(range(args.num_batches), desc="sampling"):
        key, subkey = jax.random.split(key)
        t0 = time.time()
        x_s, c_s = draw(subkey)
        x_s.block_until_ready()
        kappa.append(np.asarray(x_s))
        theta.append(np.asarray(c_s))
        # flush: stdout is block-buffered into a slurm log, so without this the
        # progress is invisible until the job ends.
        print(f"batch {i}: {time.time() - t0:.1f}s ({args.batch_size} draws)", flush=True)
    kappa = np.concatenate(kappa)
    theta = np.concatenate(theta)

    if not np.isfinite(kappa).all() or not np.isfinite(theta).all():
        print("WARNING: non-finite samples present; the CG solve or cov_t may have diverged")

    np.savez(
        out,
        observation=np.asarray(obs),
        theta_truth=np.asarray(ref["theta"]),
        mcmc_samples=np.asarray(mcmc_samples),
        reference_field=reference_field(jax.random.key(0)),
        kappa_samples=kappa * FIELD_STD + FIELD_MEAN,
        theta_samples=theta / scale_cosmo * THETA_STD + THETA_MEAN,
        method=args.method,
    )
    print(f"wrote {out}  ({kappa.shape[0]} posterior draws)")


if __name__ == "__main__":
    main()
