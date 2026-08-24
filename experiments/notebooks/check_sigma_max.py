"""Sanity-check sigma_max for the EDM diffusion config.

Draws 100 normalized kappa maps from the training dataset and reports
per-channel statistics, comparing the data's intrinsic scale to sigma_max.

For VE diffusion to make sense at sigma_max, the noise must dominate the
signal: max|x_norm| / sigma_max should be small (~<<1) so that
x_t = x + sigma_max * z ≈ sigma_max * z (pure noise).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jade.init import FIELD_MEAN, FIELD_STD

DATASET_PATH = "/u/bremy/repos/jade/experiments/sbi_lens_full"
N_MAPS = 100
SIGMA_MAX = 10.0


def main():
    ds = load_from_disk(DATASET_PATH)
    keep = [c for c in ("map",) if c in ds.column_names]
    ds = ds.select_columns(keep).with_format("numpy")
    print(f"Dataset size: {len(ds)}; columns: {ds.column_names}")

    # Draw the first N_MAPS samples (no shuffle — distribution stats are stable)
    maps = np.stack([ds[i]["map"] for i in range(N_MAPS)])  # (N, 128, 128, 5)
    print(f"Raw maps shape: {maps.shape}, dtype: {maps.dtype}")

    # Apply training normalization: (x - FIELD_MEAN) / FIELD_STD
    f_mean = np.asarray(FIELD_MEAN).reshape(1, 1, 1, -1)
    f_std = np.asarray(FIELD_STD).reshape(1, 1, 1, -1)
    x = (maps - f_mean) / f_std  # standardized per channel

    print("\nFIELD_MEAN per ch:", np.asarray(FIELD_MEAN))
    print("FIELD_STD  per ch:", np.asarray(FIELD_STD))

    print("\n--- Per-channel stats AFTER normalization (N=100 maps) ---")
    print(f"{'ch':<3}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}"
          f"{'|x|p99':>10}{'|x|p99.9':>11}{'max/σmax':>11}")
    for c in range(x.shape[-1]):
        xc = x[..., c]
        absxc = np.abs(xc)
        print(f"{c:<3}{xc.mean():>10.4f}{xc.std():>10.4f}"
              f"{xc.min():>10.4f}{xc.max():>10.4f}"
              f"{np.percentile(absxc, 99):>10.4f}"
              f"{np.percentile(absxc, 99.9):>11.4f}"
              f"{xc.max() / SIGMA_MAX:>11.4f}")

    print("\n--- Pooled stats across all channels ---")
    absx = np.abs(x)
    print(f"global std    : {x.std():.4f}")
    print(f"global |x| p99: {np.percentile(absx, 99):.4f}")
    print(f"global |x| max: {absx.max():.4f}")
    print(f"sigma_max     : {SIGMA_MAX}")
    print(f"signal/sigma_max ratio (max|x|/σmax): {absx.max()/SIGMA_MAX:.4f}")

    # Karras's t=sigma_max condition: x_T ≈ N(0, σ_max² I) requires σ_max >> max|x|.
    # Rule of thumb: σ_max >= 3 * std(x) is usually fine; the EDM defaults
    # σ_max=80 assume std(x)~1 and want a big margin.
    print(f"\nRule of thumb: σ_max >= ~3·std(x). Here 3·std = {3*x.std():.3f}")
    print(f"Sampling init x_T = σ_max · z, so var(x_T) = σ_max² = {SIGMA_MAX**2}")
    print(f"Whereas data var ≈ 1 (per channel), so the noise-to-signal var ratio"
          f" at t=T is {SIGMA_MAX**2:.1f}.")

    # Per-sample sanity: simulate x + sigma_max * z for one map, look at residual
    rng = np.random.default_rng(0)
    z = rng.standard_normal(x.shape)
    x_T = x + SIGMA_MAX * z
    snr = np.var(x, axis=(0, 1, 2)) / np.var(SIGMA_MAX * z, axis=(0, 1, 2))
    print(f"\nPer-channel SNR at t=T (var(x)/var(σ_max·z)): {snr}")
    print(f"(values much less than 1 mean noise dominates — what we want at t=T)")


if __name__ == "__main__":
    main()
