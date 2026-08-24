"""Draw the joint posterior for the single observation the reference MCMC chain
used, and write the arrays plot_paper_figures.py turns into figures.

The observation is read from the MCMC reference pickle rather than redrawn: the
diffusion posterior and the NUTS chain must condition on the same y, or the
contours describe two different posteriors.
"""

import argparse
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

import jax_cosmo as jc
from numpyro.handlers import condition, seed, trace
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal

from jade.flow import Denoiser
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.nn_hybrid import JADE_B_16
from jade.paths import FIGURES_DIR, MCMC_REF_DIR, checkpoint_dir
from jade.sampling import HeunSampler
from jade.utils import load_model

PARAMS = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]
N_PIX = 128
MAP_SIZE = 5


def build_model(ckpt, tag):
    cfg, states = load_model(ckpt, tag)
    model = JADE_B_16(
        rngs=nnx.Rngs(cfg["training"]["seed"]),
        in_channels=cfg["model"]["in_channels"],
        input_size=cfg["model"]["input_size"],
        enable_cond_image=cfg["model"]["enable_cond_image"],
        cond_channels=cfg["model"]["cond_channels"],
        num_cosmo_tokens=cfg["model"]["num_cosmo_tokens"],
        cond_patch_size=cfg["model"]["cond_patch_size"],
        cond_start=cfg["model"]["cond_start"],
        attn_drop=cfg["model"]["attn_drop"],
        proj_drop=cfg["model"]["proj_drop"],
        split_qkv=cfg["model"].get("split_qkv", False),
        mask_theta_to_field=cfg["model"].get("mask_theta_to_field", False),
    )
    model = Denoiser(model, cfg)
    nnx.update(model, states)
    return cfg, states, model


def reference_field(key):
    """A noiseless convergence field at Planck15, for the figure's colour scale
    and its 2-point reference."""
    cosmo = jc.parameters.Planck15()
    simulator = partial(
        lensingLogNormal,
        N=N_PIX,
        map_size=MAP_SIZE,
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
    tr = trace(model).get_trace()
    return np.asarray(tr["y"]["value"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(checkpoint_dir()))
    parser.add_argument("--ckpt-tag", default="JADE_B_16_latest")
    parser.add_argument("--num-steps", type=int, default=200,
                        help="Heun steps; each costs 2 network evaluations.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcmc-dir", default=str(MCMC_REF_DIR))
    parser.add_argument("--out", default=str(FIGURES_DIR / "posterior.npz"))
    args = parser.parse_args()

    print(jax.devices())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg, states, model = build_model(args.ckpt, args.ckpt_tag)
    scale_cosmo = cfg["loss"]["SCALE_COSMO"]
    print(f"t_eps={model.t_eps}  SCALE_COSMO={scale_cosmo}")

    with open(os.path.join(args.mcmc_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
        ref = pickle.load(f)
    with open(os.path.join(args.mcmc_dir,
                           "mcmc_log_posterior_samples.pkl"), "rb") as f:
        mcmc_samples = pickle.load(f)

    obs = ref["y"]
    cond = (obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)
    sampler = HeunSampler(model=model, num_steps=args.num_steps)

    def draw(key):
        keys = jax.random.split(key, 3)
        x_0 = jax.random.normal(keys[0], shape=(args.batch_size, N_PIX, N_PIX, 5))
        cosmo_0 = jax.random.normal(keys[1], shape=(args.batch_size, 6))
        return jax.vmap(sampler, in_axes=(0, 0, None, 0))(
            x_0, cosmo_0, cond, jax.random.split(keys[2], args.batch_size)
        )

    # Key sequence matches the original script exactly: the first batch uses the
    # root key itself, later batches use successive splits. Changing this would
    # give a statistically equivalent but not bit-identical set of draws.
    key = jax.random.key(args.seed)
    kappa, theta = [], []
    for i in tqdm(range(args.num_batches), desc="sampling"):
        if i == 0:
            subkey = key
        else:
            key, subkey = jax.random.split(key)
        x_s, c_s = draw(subkey)
        kappa.append(x_s)
        theta.append(c_s)
    kappa = jnp.concatenate(kappa)
    theta = jnp.concatenate(theta)

    np.savez(
        args.out,
        observation=np.asarray(obs),
        theta_truth=np.asarray(ref["theta"]),
        mcmc_samples=np.asarray(mcmc_samples),
        reference_field=reference_field(jax.random.key(0)),
        kappa_samples=np.asarray(kappa) * FIELD_STD + FIELD_MEAN,
        theta_samples=np.asarray(theta) / scale_cosmo * THETA_STD + THETA_MEAN,
    )
    print(f"wrote {args.out}  ({kappa.shape[0]} posterior draws)")


if __name__ == "__main__":
    main()
