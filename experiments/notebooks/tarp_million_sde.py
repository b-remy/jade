"""Posterior sampling for TARP using a STOCHASTIC (SDE) sampler instead of the
deterministic Heun ODE used in tarp_million.py.

The frozen model is selected with --model (see MODELS registry): `million`
(fk49rnft, 1e6 samples) or `former` (7hnur00g, the 100k model from
plot_amortized.py). Either is sampled with the Adjoint-Matching generative SDE
(Domingo-Enrich et al. 2024, Eqs. 10-11), memoryless schedule, with a
stochasticity knob g = --noise-scale:

    b(x,t)   = (1+g) v(x,t) - g * x / t
    sigma(t) = sqrt(2 g (1-t)/t)

g=0 -> deterministic Euler ODE, g=1 -> memoryless SDE. Sweep g and pick the
value whose TARP coverage sits closest to the diagonal. Output goes to a
per-g directory so different g runs never collide.
"""

import os
import argparse

import jax
import jax.numpy as jnp

from flax import nnx
import numpy as np

from jade.nn_hybrid import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.flow import Denoiser
from jade.utils import load_model

from datasets import load_from_disk
from tqdm import tqdm

# LSST noise level
from jade.init import sigma_lsst

# Model registry, switched via --model. Each entry pins the checkpoint, the
# state to load, the dataset to draw the held-out test split from, and a tag
# used for the default output directory (tarp_results/<tag>_sde_g<g>).
#   million : fk49rnft, trained on 1e6 samples (sbi_lens_million_full).
#   former  : 7hnur00g, the 100k-sample model used in plot_amortized.py /
#             notebooks/amortized_sde_1000 (state JADE_B_16_latest).
MODELS = {
    "million": {
        "ckpt": "/u/bremy/repos/jade/experiments/wandb/run-20260615_153845-fk49rnft/files/checkpoints",
        "state": "JADE_B_16_ema_best",
        "dataset": "/work/hdd/benb/bremy/sbi_lens_million_full",
        "tag": "million",
    },
    "former": {
        "ckpt": "/u/bremy/repos/jade/experiments/wandb/run-20260507_170014-7hnur00g/files/checkpoints",
        "state": "JADE_B_16_latest",
        "dataset": "/work/nvme/benb/bremy/sbi_lens_full",
        "tag": "former",
    },
}


class AdjointMatchingSDESampler(nnx.Module):
    r"""Memoryless-schedule generative SDE sampler for the linear-interpolant flow.

    Integrates ``dX = b dt + sigma dB`` forward in t from 0 (noise) to 1 (data)
    with Euler-Maruyama, where (knob ``noise_scale = g``):

        b(x,t)   = (1+g) v(x,t) - g * x / t
        sigma(t) = sqrt(2 g (1-t)/t)

    ``g=1`` is the memoryless schedule; ``g=0`` recovers the deterministic Euler
    ODE. The ``1/t`` factor is clipped with ``t_eps``; the ``1/(1-t)`` of the
    score cancels in the memoryless drift and is not clipped.

    Copied verbatim from plot_amortized_sde.py (which runs heavy model-loading
    code at import, so it cannot be imported here).
    """

    def __init__(self, model, num_steps: int = 1000, t_eps: float = 0.05,
                 noise_scale: float = 1.0):
        self.model = model
        self.num_steps = num_steps
        self.t_eps = t_eps
        self.g = noise_scale

    def __call__(self, x0, cosmo0, cond=None, key=None):
        ts = jnp.linspace(0.0, 1.0, self.num_steps + 1)
        keys = jax.random.split(key, self.num_steps)
        g = self.g

        def em_step(carry, inp):
            xt, cosmot = carry
            t, t_next, k = inp
            dt = t_next - t

            v_x, v_cosmo = self.model.v_pred(xt, cosmot, t, cond=cond, train=False)

            # Clip ONLY the diverging 1/t at t -> 0.
            t_lo = jnp.clip(t, a_min=self.t_eps)
            kappa = 1.0 / t_lo
            eta = (1.0 - t) / t_lo                 # (1-t)/t ; -> 0 cleanly at t=1
            sigma = jnp.sqrt(jnp.clip(2.0 * g * eta, a_min=0.0))

            # Memoryless drift b = (1+g) v - g kappa x  (= 2v - x/t at g=1)
            b_x = (1.0 + g) * v_x - g * kappa * xt
            b_cosmo = (1.0 + g) * v_cosmo - g * kappa * cosmot

            k_x, k_c = jax.random.split(k, 2)
            noise_x = jax.random.normal(k_x, shape=xt.shape)
            noise_cosmo = jax.random.normal(k_c, shape=cosmot.shape)

            sqrt_dt = jnp.sqrt(jnp.clip(dt, a_min=0.0))
            xt_n = xt + b_x * dt + sigma * sqrt_dt * noise_x
            cosmot_n = cosmot + b_cosmo * dt + sigma * sqrt_dt * noise_cosmo
            return (xt_n, cosmot_n), None

        (xt, cosmot), _ = jax.lax.scan(
            em_step, (x0, cosmo0), (ts[:-1], ts[1:], keys)
        )
        return xt, cosmot


def main():
    parser = argparse.ArgumentParser(description="SDE-sampler posterior sampling for TARP")
    parser.add_argument("--model", choices=list(MODELS), default="million",
                        help="Which frozen model to sample (see MODELS registry).")
    parser.add_argument("--noise-scale", type=float, required=True,
                        help="Stochasticity knob g (0=ODE, 1=memoryless SDE).")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Euler-Maruyama steps (SDE needs many more than the ODE).")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-obs", type=int, default=100,
                        help="Observations covered by THIS run, sharded across the "
                             "array jobs (obs [obs_start : obs_start+num_obs]).")
    parser.add_argument("--obs-start", type=int, default=0,
                        help="Global offset into the test set, so a later run can "
                             "add more obs without redoing the first ones.")
    parser.add_argument("--job-offset", type=int, default=0,
                        help="Added to the array task id for output filenames and "
                             "the PRNG stream, so additional runs drop in alongside "
                             "existing job_* shards in the same out_dir.")
    parser.add_argument("--out-dir", default=None,
                        help="Default: tarp_results/<model-tag>_sde_g<g>.")
    args = parser.parse_args()

    spec = MODELS[args.model]
    g = args.noise_scale
    num_samples = args.num_samples
    out_dir = args.out_dir or f"tarp_results/{spec['tag']}_sde_g{g}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"model={args.model} ({spec['state']})  noise_scale g={g}  "
          f"num_steps={args.num_steps}  out_dir={out_dir}")

    cfg, states = load_model(spec["ckpt"], spec["state"])

    model = JADE_B_16(
        rngs=nnx.Rngs(cfg['training']['seed']),
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

    model = Denoiser(model, cfg)
    nnx.update(model, states)

    # Clip for the 1/(1-t) inside v_pred (matches plot_amortized_sde.py).
    model.t_eps = 0.05

    sampler = AdjointMatchingSDESampler(
        model=model, num_steps=args.num_steps, t_eps=0.05, noise_scale=g
    )

    @jax.jit
    def make_cond(x, key):
        cond = x + sigma_lsst * jax.random.normal(key, shape=x.shape)
        cond = (cond - FIELD_MEAN.reshape(1, 1, 1, -1)) / FIELD_STD.reshape(1, 1, 1, -1)
        return cond

    @jax.jit
    def posterior_sampling(cond_i, key):
        keys = jax.random.split(key, 3)

        x_0 = jax.random.normal(keys[0], shape=(num_samples, 128, 128, 5))
        cosmo_0 = jax.random.normal(keys[1], shape=(num_samples, 6))

        x_samples, cosmo_samples = jax.vmap(sampler, in_axes=(0, 0, None, 0))(
            x_0, cosmo_0, cond_i, jax.random.split(keys[2], num_samples)
        )
        x_samples = x_samples * FIELD_STD + FIELD_MEAN
        cosmo_samples = cosmo_samples * THETA_STD + THETA_MEAN
        return x_samples, cosmo_samples

    # Held-out test split, reproducing fk49rnft's exact train/test split.
    dataset = load_from_disk(spec["dataset"])
    dataset = dataset.train_test_split(
        test_size=cfg['data']['val_split'],
        seed=cfg['data']['shuffle_seed'],
    )["test"]
    dataset = dataset.with_format("numpy")

    job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    n_jobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    print(f"Starting job {job_id} out of {n_jobs}")

    # Take the first num_obs test observations and SHARD them across the array
    # jobs (e.g. 100 obs / 4 jobs = 25 obs each), so the whole array produces
    # num_obs posteriors TOTAL — not num_obs per job. Last job takes any
    # remainder.
    num_obs = args.num_obs
    obs_start = args.obs_start
    full = dataset[obs_start:obs_start + num_obs]
    maps_full = np.asarray(full["map"])
    theta_full = np.asarray(full["theta"])

    per = num_obs // n_jobs
    start = job_id * per
    end = num_obs if job_id == n_jobs - 1 else start + per
    maps = maps_full[start:end]
    theta = theta_full[start:end]
    n_obs = len(maps)

    # Global job index: unique across separate runs that share an out_dir, so
    # additional obs drop in alongside existing job_* shards (filenames + seed).
    gjid = args.job_offset + job_id
    print(f"job {job_id} (global {gjid}): obs "
          f"[{obs_start + start}:{obs_start + end}] -> {n_obs} observations")

    # Independent PRNG stream per global job index.
    key = jax.random.fold_in(jax.random.PRNGKey(0), gjid)

    # Stream the huge field posterior straight to a disk-backed .npy (memmap),
    # as in tarp_million.py, so peak host RAM stays ~1-2 GB.
    x_mm = np.lib.format.open_memmap(
        f'{out_dir}/x_samples_job_{gjid}.npy',
        mode='w+', dtype=np.float32,
        shape=(n_obs, num_samples, 128, 128, 5),
    )
    theta_prior = np.asarray(theta, dtype=np.float32)
    theta_posterior = np.empty((n_obs, num_samples, 6), dtype=np.float32)

    key, sk = jax.random.split(key)
    cond = make_cond(maps, sk)

    for i in tqdm(range(n_obs)):
        key, sk = jax.random.split(key)
        x_samples, cosmo_samples = posterior_sampling(cond[i], sk)
        x_mm[i] = np.asarray(x_samples, dtype=np.float32)
        theta_posterior[i] = np.asarray(cosmo_samples, dtype=np.float32)

    x_mm.flush()
    del x_mm
    np.save(f'{out_dir}/true_cosmo_job_{gjid}.npy', theta_prior)
    np.save(f'{out_dir}/cosmo_samples_job_{gjid}.npy', theta_posterior)
    np.save(f'{out_dir}/true_x_job_{gjid}.npy', np.asarray(maps))


if __name__ == "__main__":
    main()
