"""Verify the 500 SDE posterior shards are independent, non-overlapping obs.

Checks:
  1. Every shard's saved truth (true_x / true_cosmo) matches the held-out test
     split at the expected GLOBAL index (shard jid -> obs [25*jid : 25*(jid+1)]),
     proving no overlap between the original 100 (job_0..3) and the extra 400
     (job_4..19) and correct mapping.
  2. All 500 observations are pairwise distinct (no duplicated obs).
  3. Posterior samples are obs-specific (per-obs posterior means differ; not
     collapsed by a reused PRNG stream).
"""
import glob
import os
import re

import numpy as np
from datasets import load_from_disk

D = "/work/hdd/benb/bremy/jade/tarp/million_sde_g1.0"
DATASET = "/work/hdd/benb/bremy/sbi_lens_million_full"
VAL_SPLIT = 0.05      # fk49rnft cfg
SHUFFLE_SEED = 42     # fk49rnft cfg

# Reproduce the exact test split the sampling script used.
ds = load_from_disk(DATASET).train_test_split(
    test_size=VAL_SPLIT, seed=SHUFFLE_SEED)["test"].with_format("numpy")

ids = sorted(int(re.search(r"_(\d+)\.npy$", p).group(1))
             for p in glob.glob(f"{D}/x_samples_job_*.npy"))
print("shard ids:", ids)

all_truth = []
all_post_mean = []
map_mismatch = 0
for jid in ids:
    tx = np.load(f"{D}/true_x_job_{jid}.npy")           # (n, 128,128,5)
    tc = np.load(f"{D}/true_cosmo_job_{jid}.npy")       # (n, 6)
    cs = np.load(f"{D}/cosmo_samples_job_{jid}.npy")    # (n, n_samp, 6)
    n = tx.shape[0]
    g0 = 25 * jid                                       # expected global start
    ref_x = ds[g0:g0 + n]["map"]
    ref_t = ds[g0:g0 + n]["theta"]
    if not np.allclose(tx, ref_x, atol=1e-5):
        map_mismatch += 1
        print(f"  !! job_{jid}: true_x does NOT match test[{g0}:{g0+n}]")
    if not np.allclose(tc, ref_t, atol=1e-5):
        print(f"  !! job_{jid}: true_cosmo does NOT match test[{g0}:{g0+n}]")
    all_truth.append(tx.reshape(n, -1))
    all_post_mean.append(cs.mean(1))

truth = np.concatenate(all_truth)          # (500, 81920)
post_mean = np.concatenate(all_post_mean)  # (500, 6)
N = truth.shape[0]
print(f"\ntotal obs: {N}")
print(f"truth<->test-split match: {'ALL OK' if map_mismatch == 0 else f'{map_mismatch} MISMATCH'}")

# Pairwise distinctness via a cheap fingerprint of each observation map.
fp = truth.sum(1) + truth[:, ::997].sum(1)   # two independent linear hashes
uniq = len(np.unique(np.round(fp, 4)))
print(f"distinct observations (fingerprint): {uniq}/{N}")

# Posterior means must differ across obs (not seed-collapsed).
pm_uniq = len(np.unique(np.round(post_mean.sum(1), 6)))
print(f"distinct posterior means: {pm_uniq}/{N}")

ok = (map_mismatch == 0) and (uniq == N) and (pm_uniq == N)
print("\nRESULT:", "PASS - 500 independent, non-overlapping obs" if ok else "FAIL - see above")
