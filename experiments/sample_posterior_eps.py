#!/usr/bin/env python
"""Posterior sampling with an Exact Posterior Score (EPS) denoiser
(arXiv:2606.17048), on the same observation and likelihood as
`sample_posterior_guidance.py`.

DPS and MMPS steer the frozen unconditional prior with an approximate likelihood
correction at every step. EPS instead uses a copy of that prior fine-tuned by
`train_eps.py` to denoise the closed-form posterior pivot
(`jade.flow_eps.EPSDenoiser`). Each sampler step is then one plain network
evaluation with no VJP or linear solve, and the only approximation left is the
network's fit.

The observation is the noisy convergence map behind the reference NUTS chain
(`mcmc_log_normal/mcmc_log_obs_truth.pkl`), mapped to shear with the same KS
inversion as the guidance script, so DPS, MMPS and EPS all target the same
posterior. Output uses the `.npz` schema of `sample_posterior.py`, so
`plot_guidance_figures.py` and `plot_paper_figures.py` read it directly.

Examples
--------
    python sample_posterior_eps.py --ckpt wandb/run-.../files/checkpoints
    python sample_posterior_eps.py --ckpt ... --sampler euler --num-steps 128
"""

import argparse
import os
import pickle
import time

import jax
import numpy as np
from sample_posterior_guidance import build_model, reference_field, shear_observation
from tqdm import tqdm

from jade.flow_eps import EPSDenoiser
from jade.init import FIELD_MEAN, FIELD_STD, THETA_MEAN, THETA_STD, sigma_lsst
from jade.paths import FIGURES_DIR, MCMC_REF_DIR
from jade.sampling import EulerSampler, HeunSampler

N_PIX = 128
N_BINS = 5
COSMO_DIM = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory of a train_eps.py run.")
    parser.add_argument("--ckpt-tag", default="JADE_B_16_ema_latest")
    parser.add_argument("--sampler", choices=["euler", "heun"], default="heun")
    # EPS uses the backbone's own sampler, so the step count is set by the ODE
    # integration error alone, as for the prior, not by guidance stability.
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcmc-dir", default=str(MCMC_REF_DIR))
    parser.add_argument("--out", default=str(FIGURES_DIR / "posterior_eps.npz"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(jax.devices())

    cfg, model = build_model(args.ckpt, args.ckpt_tag)
    # A plain prior checkpoint has the same layout, and would run and return
    # samples of the wrong distribution, silently.
    if "eps" not in cfg:
        raise ValueError(
            f"{args.ckpt} is not an EPS checkpoint (no `eps` block in its config); train with train_eps.py."
        )
    scale_cosmo = cfg["loss"]["SCALE_COSMO"]

    with open(os.path.join(args.mcmc_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
        ref = pickle.load(f)
    with open(os.path.join(args.mcmc_dir, "mcmc_log_posterior_samples.pkl"), "rb") as f:
        mcmc_samples = pickle.load(f)

    obs = ref["y"]
    gamma = shear_observation(obs)
    print(f"observation {np.shape(obs)} -> shear {gamma.shape}")

    pdenoiser = EPSDenoiser(model=model, cfg=cfg, gamma=gamma, sigma_gamma=sigma_lsst)
    sampler_cls = EulerSampler if args.sampler == "euler" else HeunSampler
    sampler = sampler_cls(model=pdenoiser, num_steps=args.num_steps)
    print(f"EPS | {args.sampler} x {args.num_steps} steps")

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
        print(f"batch {i}: {time.time() - t0:.1f}s ({args.batch_size} draws)", flush=True)
    kappa = np.concatenate(kappa)
    theta = np.concatenate(theta)

    if not np.isfinite(kappa).all() or not np.isfinite(theta).all():
        print("WARNING: non-finite samples present")

    np.savez(
        args.out,
        observation=np.asarray(obs),
        theta_truth=np.asarray(ref["theta"]),
        mcmc_samples=np.asarray(mcmc_samples),
        reference_field=reference_field(jax.random.key(0)),
        kappa_samples=kappa * FIELD_STD + FIELD_MEAN,
        theta_samples=theta / scale_cosmo * THETA_STD + THETA_MEAN,
        method="eps",
    )
    print(f"wrote {args.out}  ({kappa.shape[0]} posterior draws)")


if __name__ == "__main__":
    main()
