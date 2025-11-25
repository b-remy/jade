"""
Generate a HuggingFace dataset of lensing maps using sbi_lens simulator.
This script generates 100,000 examples.
"""

import jax
import jax.numpy as jnp
from functools import partial
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import Dataset, Features, Array3D, Sequence, Value

import argparse

from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
# from sbi_lens.simulator.utils import get_samples_and_scores
from utils import get_samples

def setup_model(N=128, map_size=5, with_noise=False):
    """Setup the lensing model with LSST Year 10 configuration."""
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
        model_type='lognormal',
        lognormal_shifts='LSSTY10',
        with_noise=with_noise,
    )
    
    return model_log_normal


def generate_batch(model, key, batch_size, with_noise=False):
    """Generate a batch of samples."""
    (_, samples)= get_samples(
        model,
        key,
        batch_size=batch_size,
        with_noise=with_noise
    )
    
    maps = samples['y']  # Shape: (batch_size, N, N) or (batch_size, nbins, N, N)
    theta = samples['theta']  # Shape: (batch_size, n_params)
    
    return maps, theta


def sample_generator(model, batch_size, total_samples, seed, with_noise=False):
    """
    Generator function that yields individual samples.
    This is memory efficient as it doesn't store all samples in memory.
    """
    key = jax.random.key(seed)
    num_batches = total_samples // batch_size

    for _ in tqdm(range(num_batches)):
        key, subkey = jax.random.split(key)
        maps, theta = generate_batch(model, subkey, batch_size, with_noise)
        
        # maps_np = np.array(maps)
        # theta_np = np.array(theta)
        
        for i in range(batch_size):
            example = {
                "map": maps[i],
                "theta": theta[i],
            }
            
            yield example


def generate_dataset(
    # total_samples=100000,
    total_samples=10,
    batch_size=10,
    output_dir="./lensing_dataset",
    N=128, # pixels
    map_size=5, # degrees
    with_noise=False,
    seed=0,
):
    """
    Generate dataset and save to disk using from_generator.
    
    Args:
        total_samples: Total number of examples to generate
        batch_size: Batch size for generation
        output_dir: Directory to save the dataset
        N: Grid resolution
        map_size: Map size in degrees
        with_noise: Whether to add noise to simulations
        seed: Random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model
    print("Setting up model...")
    model = setup_model(N=N, map_size=map_size, with_noise=with_noise)
    
    print(f"Using batch_size={batch_size}")
    
    # Generate one sample to determine shapes and parameter names
    print("Determining data structure...")
    test_key = jax.random.key(seed)
    test_maps, test_theta = generate_batch(model, test_key, 1, with_noise)
    
    map_shape = test_maps.shape[1:]  # Remove batch dimension
    n_params = test_theta.shape[1]
    
    print(f"Map shape: {map_shape}")
    print(f"Number of theta parameters: {n_params}")
    print(f"Generating {total_samples} samples in batches of {batch_size}...")
    
    # Define dataset features
    features = Features({
        "map": Array3D(dtype="float32", shape=map_shape),
        "theta": Sequence(Value(dtype="float32"), length=n_params),
    })
    
    # Create generator
    gen = lambda: sample_generator(model, batch_size, total_samples, seed, with_noise)
    
    # Create dataset from generator
    print("\nCreating dataset from generator (this is memory efficient)...")
    dataset = Dataset.from_generator(
        gen,
        features=features,
        cache_dir=str(output_dir / "cache"),
    )
    
    print(f"\n✓ Dataset generation complete!")
    print(f"  Total samples: {len(dataset)}")
    
    # Save to disk
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))
    
    print(f"  Saved to: {output_dir}")
    print(f"\nTo load the dataset:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_dir}')")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate lensing map dataset")
    parser.add_argument("--total-samples", type=int, default=100000,
                        help="Total number of samples to generate")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Batch size for generation")
    parser.add_argument("--output-dir", type=str, default="./lensing_dataset",
                        help="Output directory for dataset")
    parser.add_argument("--N", type=int, default=128,
                        help="Grid resolution")
    parser.add_argument("--map-size", type=float, default=5.0,
                        help="Map size in degrees")
    parser.add_argument("--with-noise", action="store_true",
                        help="Add noise to simulations")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    args = parser.parse_args()
    
    dataset = generate_dataset(
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        N=args.N,
        map_size=args.map_size,
        with_noise=args.with_noise,
        seed=args.seed,
    )
    
    # Print some statistics
    print("\nDataset statistics:")
    print(dataset)


if __name__ == "__main__":
    main()