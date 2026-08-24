"""Mira calibration score on the posterior samples written by
sample_calibration.py.

``--space cosmo`` scores the cosmology-marginal posterior p(theta | y),
``--space joint`` the full p(theta, kappa | y) with the flattened field
concatenated after the 6 cosmology dimensions. The joint tensor is ~76 GiB and
wants a GPU.
"""

import argparse
import glob
import os
import re
import time

import numpy as np
import torch

from mira_score import get_device, mira_bootstrap_efficient

from jade.paths import RESULTS_DIR


def _job_ids(samples_dir, space):
    """Job ids whose sample files are all present."""
    stem = "cosmo_samples_job_" if space == "cosmo" else "x_samples_job_"
    files = sorted(
        glob.glob(os.path.join(samples_dir, stem + "*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    needed = ["cosmo_samples_job_{}.npy", "true_cosmo_job_{}.npy"]
    if space == "joint":
        needed += ["x_samples_job_{}.npy", "true_x_job_{}.npy"]

    ids = []
    for f in files:
        jid = int(re.search(r"_(\d+)\.npy$", f).group(1))
        if all(os.path.exists(os.path.join(samples_dir, n.format(jid)))
               for n in needed):
            ids.append(jid)
    return ids


def load_to_gpu(samples_dir, space, device, max_jobs=None):
    """Posterior samples as (1, T, S, q) and truths as (T, q)."""
    ids = _job_ids(samples_dir, space)
    if max_jobs is not None:
        ids = ids[:max_jobs]

    c0 = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{ids[0]}.npy"),
                 mmap_mode="r")
    obs_per_job, S, q_cosmo = c0.shape
    del c0

    q_field = 0
    if space == "joint":
        x0 = np.load(os.path.join(samples_dir, f"x_samples_job_{ids[0]}.npy"),
                     mmap_mode="r")
        q_field = int(np.prod(x0.shape[2:]))
        del x0

    q_total = q_cosmo + q_field
    T_total = obs_per_job * len(ids)

    print(f"Allocating {space} posterior on {device}: "
          f"(1, {T_total}, {S}, {q_total}) float32 = "
          f"{T_total * S * q_total * 4 / 2**30:.3f} GiB "
          f"(q_cosmo={q_cosmo}, q_field={q_field})")
    posterior_gpu = torch.empty((1, T_total, S, q_total),
                                dtype=torch.float32, device=device)
    truth_gpu = torch.empty((T_total, q_total),
                            dtype=torch.float32, device=device)

    for jid in ids:
        t0 = time.time()
        start = jid * obs_per_job
        end = start + obs_per_job

        c = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{jid}.npy"))
        tc = np.load(os.path.join(samples_dir, f"true_cosmo_job_{jid}.npy"))
        posterior_gpu[0, start:end, :, :q_cosmo].copy_(torch.from_numpy(c))
        truth_gpu[start:end, :q_cosmo].copy_(torch.from_numpy(tc))
        del c, tc

        if space == "joint":
            x = np.load(os.path.join(samples_dir, f"x_samples_job_{jid}.npy"))
            tx = np.load(os.path.join(samples_dir, f"true_x_job_{jid}.npy"))
            posterior_gpu[0, start:end, :, q_cosmo:].copy_(
                torch.from_numpy(x.reshape(obs_per_job, S, q_field))
            )
            truth_gpu[start:end, q_cosmo:].copy_(
                torch.from_numpy(tx.reshape(obs_per_job, q_field))
            )
            del x, tx

        print(f"  job {jid}: {space} -> GPU ({time.time() - t0:.1f}s)")

    return posterior_gpu, truth_gpu, q_cosmo, q_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_dir", nargs="?",
                        default=str(RESULTS_DIR / "conditional"))
    parser.add_argument("--space", choices=["cosmo", "joint"], default="joint")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=1,
                        help="MC runs per bootstrap iter (mira_bootstrap "
                             "reference uses 1).")
    parser.add_argument("--save-npz", default=None,
                        help="Defaults to mira_<space>_scores.npz next to the "
                             "samples' parent directory.")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    save_npz = args.save_npz or str(RESULTS_DIR / f"mira_{args.space}_scores.npz")

    device = get_device()
    print(f"device: {device}")

    t0 = time.time()
    posterior_gpu, truth_gpu, q_cosmo, q_field = load_to_gpu(
        args.samples_dir, args.space, device, max_jobs=args.max_jobs
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
    print(f"mira {args.space} score: {m:.6f} +/- {s:.6f}   "
          f"(wall {t:.1f}s, {args.num_bootstrap} boots × num_runs={args.num_runs})")
    print(f"SE = {s / np.sqrt(args.num_bootstrap):.6f}")
    np.savez(save_npz, mean=m, std=s,
             q_cosmo=q_cosmo, q_field=q_field, norm=args.norm)
    print(f"wrote {save_npz}")


if __name__ == "__main__":
    main()
