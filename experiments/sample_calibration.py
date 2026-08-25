"""Posterior samples over held-out observations, for plot_tarp.py and
plot_mira.py.

One SLURM array task per chunk of observations; each writes
{true_cosmo,cosmo_samples,true_x,x_samples}_job_<id>.npy. The paper used 5 tasks
of 100 observations with 500 draws each.
"""

import argparse
import os

import jax
import numpy as np
from datasets import load_from_disk
from flax import nnx
from tqdm import tqdm

from jade.flow import Denoiser
from jade.init import FIELD_MEAN, FIELD_STD, THETA_MEAN, THETA_STD, sigma_lsst
from jade.nn import JADE_B_16
from jade.paths import DATASET_DIR, RESULTS_DIR, checkpoint_dir
from jade.sampling import HeunSampler
from jade.utils import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(checkpoint_dir()))
    parser.add_argument("--ckpt-tag", default="JADE_B_16_ema_best")
    parser.add_argument("--num-steps", type=int, default=128, help="Heun steps; each costs 2 network evaluations.")
    parser.add_argument("--num-samples", type=int, default=500, help="Posterior draws per observation.")
    parser.add_argument("--batch-size", type=int, default=100, help="Observations per array task.")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "conditional"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg, states = load_model(args.ckpt, args.ckpt_tag)

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
        # Stage-2 runs set this; older stage-1 configs don't have the key.
        split_qkv=cfg["model"].get("split_qkv", False),
        mask_theta_to_field=cfg["model"].get("mask_theta_to_field", False),
    )

    model = Denoiser(model, cfg)
    nnx.update(model, states)

    # Built once so it is traced/compiled exactly once when posterior_sampling
    # is first called — never per-condition or per-loop-step.
    sampler = HeunSampler(model=model, num_steps=args.num_steps)
    num_samples = args.num_samples

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

    # Held-out test split only, reproducing the exact train/test split used
    # during training (see train.py: train_test_split(test_size=val_split,
    # seed=shuffle_seed)). Iterating the full dataset here would draw ~95%
    # in-sample observations and make the coverage look artificially good.
    dataset = load_from_disk(args.dataset)
    dataset = dataset.train_test_split(
        test_size=cfg["data"]["val_split"],
        seed=cfg["data"]["shuffle_seed"],
    )["test"]
    dataset = dataset.with_format("numpy")

    loader = dataset.iter(batch_size=args.batch_size)

    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 5))

    print(f"Starting job {job_id} out of {n_jobs}")

    key = jax.random.PRNGKey(0)
    data_batch = next(loader)
    for _ in range(job_id):
        data_batch = next(loader)
        key, _ = jax.random.split(key)

    theta_prior = []
    theta_posterior = []
    x_posterior = []

    key, sk = jax.random.split(key)
    cond = make_cond(data_batch["map"], sk)

    for i in tqdm(range(len(data_batch["map"]))):
        key, sk = jax.random.split(key)
        x_samples, cosmo_samples = posterior_sampling(cond[i], sk)
        theta_prior.append(data_batch["theta"][i])
        theta_posterior.append(cosmo_samples)
        x_posterior.append(np.array(x_samples))

    theta_prior = np.array(theta_prior)
    theta_posterior = np.array(theta_posterior)
    x_posterior = np.array(x_posterior)
    for name, array in [
        ("true_cosmo", theta_prior),
        ("cosmo_samples", theta_posterior),
        ("true_x", data_batch["map"]),
        ("x_samples", x_posterior),
    ]:
        path = os.path.join(args.out_dir, f"{name}_job_{job_id}.npy")
        np.save(path, array)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
