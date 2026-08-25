"""
Generate a HuggingFace dataset of lensing maps using sbi_lens simulator.
Supports parallel independent jobs via --job-id using JAX fold_in.
"""

import argparse
from functools import partial
from pathlib import Path

import jax
from datasets import Array3D, Dataset, Features, Sequence, Value
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from tqdm import tqdm
from utils import get_samples

# ------------------------------------------------------------------
# Model setup
# ------------------------------------------------------------------


def setup_model(N=128, map_size=5, with_noise=False):
    sigma_e = config_lsst_y_10.sigma_e
    gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
    nbins = config_lsst_y_10.nbins
    a = config_lsst_y_10.a
    b = config_lsst_y_10.b
    z0 = config_lsst_y_10.z0

    model_log_normal = partial(
        lensingLogNormal,
        N=N,
        map_size=map_size,
        gal_per_arcmin2=gals_per_arcmin2,
        sigma_e=sigma_e,
        nbins=nbins,
        a=a,
        b=b,
        z0=z0,
        model_type="lognormal",
        lognormal_shifts="LSSTY10",
        with_noise=with_noise,
    )

    return model_log_normal


# ------------------------------------------------------------------
# Batch generation
# ------------------------------------------------------------------


def make_sample_fn(model, batch_size, with_noise=False):
    """Build the jitted sampling function once so it is reused (compiled a
    single time) across all batches, instead of recompiling per batch."""
    return jax.jit(
        partial(
            get_samples,
            model=model,
            batch_size=batch_size,
            with_noise=with_noise,
        )
    )


def generate_batch(sample_fn, key):
    _, samples = sample_fn(key=key)

    maps = samples["y"]  # convergence map (kappa), [batch_size, N, N, nbins]
    theta = samples["theta"]

    return maps, theta


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------


def sample_generator(
    N,
    map_size,
    with_noise,
    batch_size,
    total_samples,
    base_seed,
    job_id,
):

    base_key = jax.random.key(base_seed)
    key = jax.random.fold_in(base_key, job_id)

    model = setup_model(N=N, map_size=map_size, with_noise=with_noise)
    sample_fn = make_sample_fn(model, batch_size, with_noise)

    num_batches = total_samples // batch_size

    for _ in tqdm(range(num_batches)):
        key, subkey = jax.random.split(key)
        maps, theta = generate_batch(sample_fn, subkey)

        for i in range(batch_size):
            yield {
                "map": maps[i],
                "theta": theta[i],
            }


# ------------------------------------------------------------------
# Dataset creation
# ------------------------------------------------------------------


def generate_dataset(
    total_samples,
    batch_size,
    output_dir,
    N=128,
    map_size=5,
    with_noise=False,
    base_seed=0,
    job_id=0,
):
    """
    Generate dataset for a given job_id using fold_in for independent randomness.
    """

    print(f"\nJob ID: {job_id}")
    print(f"Base seed: {base_seed}")
    print("Using fold_in → independent PRNG stream\n")

    output_dir = Path(output_dir) / f"job_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up model...")
    model = setup_model(N=N, map_size=map_size, with_noise=with_noise)

    print("Determining data structure...")
    # test_key, _ = jax.random.split(job_key)
    test_key = jax.random.key(0)
    test_sample_fn = make_sample_fn(model, 1, with_noise)
    test_maps, test_theta = generate_batch(test_sample_fn, test_key)

    map_shape = test_maps.shape[1:]
    n_params = test_theta.shape[1]

    features = Features(
        {
            "map": Array3D(dtype="float32", shape=map_shape),
            "theta": Sequence(Value(dtype="float32"), length=n_params),
        }
    )

    # gen = lambda: sample_generator(
    #     model,
    #     batch_size,
    #     total_samples,
    #     base_seed,
    #     job_id,
    #     with_noise,
    # )

    # print("Creating dataset from generator...")
    # dataset = Dataset.from_generator(
    #     gen,
    #     features=features,
    #     cache_dir=str(output_dir / "cache"),
    # )

    dataset = Dataset.from_generator(
        sample_generator,
        gen_kwargs={
            "N": N,
            "map_size": map_size,
            "with_noise": with_noise,
            "batch_size": batch_size,
            "total_samples": total_samples,
            "base_seed": base_seed,
            "job_id": job_id,
        },
        features=features,
        cache_dir=str(output_dir / "cache"),
    )

    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))

    print(f"✓ Job {job_id} complete. Samples: {len(dataset)}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate lensing map dataset")

    parser.add_argument("--total-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="./lensing_dataset")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--map-size", type=float, default=5.0)
    parser.add_argument("--with-noise", action="store_true")

    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--job-id", type=int, required=True, help="Parallel job id (0,1,2,...)")

    args = parser.parse_args()

    generate_dataset(
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        N=args.N,
        map_size=args.map_size,
        with_noise=args.with_noise,
        base_seed=args.seed,
        job_id=args.job_id,
    )


if __name__ == "__main__":
    main()
