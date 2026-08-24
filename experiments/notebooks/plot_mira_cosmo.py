"""Cosmology-marginal θ Mira score on the posterior samples written by
tarp_conditional.py — cosmo-only analog of plot_mira_joint.py.

Loads only the cosmo samples per job into one GPU tensor of shape
(1, T, S, q_cosmo), then calls :func:`mira_score.mira_bootstrap_efficient`.
"""

import argparse
import glob
import os
import re
import time

import numpy as np
import torch

from mira_score import get_device, mira_bootstrap_efficient


def _job_ids(samples_dir):
    c_files = sorted(
        glob.glob(os.path.join(samples_dir, "cosmo_samples_job_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    ids = []
    for cf in c_files:
        jid = int(re.search(r"_(\d+)\.npy$", cf).group(1))
        needed = [
            os.path.join(samples_dir, f"cosmo_samples_job_{jid}.npy"),
            os.path.join(samples_dir, f"true_cosmo_job_{jid}.npy"),
        ]
        if all(os.path.exists(p) for p in needed):
            ids.append(jid)
    return ids


def load_cosmo_to_gpu(samples_dir, device, max_jobs=None):
    ids = _job_ids(samples_dir)
    if max_jobs is not None:
        ids = ids[:max_jobs]
    c0 = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{ids[0]}.npy"),
                 mmap_mode="r")
    obs_per_job, S, q_cosmo = c0.shape
    T_total = obs_per_job * len(ids)
    del c0

    print(f"Allocating cosmo posterior on {device}: "
          f"(1, {T_total}, {S}, {q_cosmo}) float32 = "
          f"{T_total * S * q_cosmo * 4 / 2**30:.3f} GiB")
    posterior_gpu = torch.empty((1, T_total, S, q_cosmo),
                                dtype=torch.float32, device=device)
    truth_gpu = torch.empty((T_total, q_cosmo),
                            dtype=torch.float32, device=device)

    for jid in ids:
        t0 = time.time()
        start = jid * obs_per_job
        end = start + obs_per_job

        c = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{jid}.npy"))
        tc = np.load(os.path.join(samples_dir, f"true_cosmo_job_{jid}.npy"))
        posterior_gpu[0, start:end].copy_(torch.from_numpy(c))
        truth_gpu[start:end].copy_(torch.from_numpy(tc))
        del c, tc
        print(f"  job {jid}: cosmo -> GPU ({time.time() - t0:.1f}s)")

    return posterior_gpu, truth_gpu, q_cosmo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_dir", nargs="?",
                        default="tarp_results/conditional")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=1,
                        help="MC runs per bootstrap iter (mira_bootstrap "
                             "reference uses 1).")
    parser.add_argument("--save-npz",
                        default="tarp_results/mira_cosmo_scores.npz")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    t0 = time.time()
    posterior_gpu, truth_gpu, q_cosmo = load_cosmo_to_gpu(
        args.samples_dir, device, max_jobs=args.max_jobs
    )
    print(f"load done in {time.time() - t0:.1f}s.")
    if device.type == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 2**30:.3f} GiB")
    print(f"normalisation: {'ON' if args.norm else 'OFF'}")

    torch.manual_seed(2024)
    t0 = time.time()
    boot_mean, boot_std = mira_bootstrap_efficient(
        truth_gpu, posterior_gpu,
        num_bootstrap=args.num_bootstrap,
        num_runs=args.num_runs, norm=args.norm,
        disable_tqdm=True, device=device,
    )
    t = time.time() - t0
    m = float(boot_mean[0].cpu()); s = float(boot_std[0].cpu())
    print(f"mira cosmo score: {m:.6f} +/- {s:.6f}   "
          f"(wall {t:.1f}s, {args.num_bootstrap} boots × num_runs={args.num_runs})")
    print(f"SE = {s / np.sqrt(args.num_bootstrap):.6f}")
    np.savez(args.save_npz, mean=m, std=s, q_cosmo=q_cosmo, norm=args.norm)
    print(f"wrote {args.save_npz}")


if __name__ == "__main__":
    main()
