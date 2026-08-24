"""Joint (κ, θ) Pokie on the posterior samples written by tarp_conditional.py.

For each observation ``i``, both ``x_samples_job_*[i, j, ...]`` (the κ field
draws) and ``cosmo_samples_job_*[i, j, ...]`` (the cosmology draws) come from
the **same** ``posterior_sampling`` call with the same RNG key
(``tarp_conditional.py:91``) — so the pair ``(x_samples[i, j], cosmo_samples[i, j])``
is a single joint posterior draw conditioned on the same observation.
Concatenating along the last axis therefore gives valid joint samples of
``p(κ, θ | obs)`` and Pokie tests joint calibration.

Layout used here: cosmology dims first, then flattened κ pixels
→ joint q = 6 + 128*128*5 = 81926. Cosmology occupies the first 6 dims,
field the rest.

Memory: posterior on GPU is (1, T, S, 81926) ≈ 76.4 GiB at float32 — six
extra columns vs the field-only run. Host peak: one field file at a time
(16.4 GB). Fits a single GH200 + 120 GB host comfortably.
"""

import argparse
import glob
import os
import re
import time

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

from pokie import get_device, pokie_bootstrap_efficient


def _job_ids(samples_dir):
    """Return sorted list of job ids for which all four files exist."""
    x_files = sorted(
        glob.glob(os.path.join(samples_dir, "x_samples_job_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    ids = []
    for xf in x_files:
        jid = int(re.search(r"_(\d+)\.npy$", xf).group(1))
        needed = [
            os.path.join(samples_dir, f"x_samples_job_{jid}.npy"),
            os.path.join(samples_dir, f"true_x_job_{jid}.npy"),
            os.path.join(samples_dir, f"cosmo_samples_job_{jid}.npy"),
            os.path.join(samples_dir, f"true_cosmo_job_{jid}.npy"),
        ]
        if all(os.path.exists(p) for p in needed):
            ids.append(jid)
        else:
            missing = [p for p in needed if not os.path.exists(p)]
            print(f"warning: skipping job {jid}, missing {missing}")
    return ids


def load_joint_to_gpu(samples_dir, device, max_jobs=None):
    """Stream-load (x, cosmo) per-job files into a single pre-allocated
    GPU tensor with cosmology occupying the first columns of the q axis.

    Returns ``(posterior_gpu, truth_gpu, q_cosmo, q_field)``.
    """
    ids = _job_ids(samples_dir)
    if not ids:
        raise FileNotFoundError(f"no usable job files in {samples_dir}")
    if max_jobs is not None:
        ids = ids[:max_jobs]
        print(f"limiting to first {len(ids)} job(s): {ids}")

    # Peek shapes.
    x0 = np.load(os.path.join(samples_dir, f"x_samples_job_{ids[0]}.npy"),
                 mmap_mode="r")
    c0 = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{ids[0]}.npy"),
                 mmap_mode="r")
    obs_per_job, S, *spatial = x0.shape  # (n_obs, S, 128, 128, 5)
    q_field = int(np.prod(spatial))
    q_cosmo = c0.shape[-1]
    q_total = q_cosmo + q_field
    T_total = obs_per_job * len(ids)
    del x0, c0

    print(f"Allocating joint posterior on {device}: "
          f"(1, {T_total}, {S}, {q_total}) float32 = "
          f"{T_total * S * q_total * 4 / 2**30:.1f} GiB  "
          f"(q_cosmo={q_cosmo}, q_field={q_field})")
    posterior_gpu = torch.empty((1, T_total, S, q_total),
                                dtype=torch.float32, device=device)
    truth_gpu = torch.empty((T_total, q_total),
                            dtype=torch.float32, device=device)

    for jid in ids:
        t0 = time.time()
        start = jid * obs_per_job
        end = start + obs_per_job

        # Cosmology slice → first columns.
        c = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{jid}.npy"))
        tc = np.load(os.path.join(samples_dir, f"true_cosmo_job_{jid}.npy"))
        posterior_gpu[0, start:end, :, :q_cosmo].copy_(torch.from_numpy(c))
        truth_gpu[start:end, :q_cosmo].copy_(torch.from_numpy(tc))
        del c, tc

        # Field slice → remaining columns.
        x = np.load(os.path.join(samples_dir, f"x_samples_job_{jid}.npy"))
        tx = np.load(os.path.join(samples_dir, f"true_x_job_{jid}.npy"))
        x_flat = x.reshape(obs_per_job, S, q_field)
        tx_flat = tx.reshape(obs_per_job, q_field)
        posterior_gpu[0, start:end, :, q_cosmo:].copy_(torch.from_numpy(x_flat))
        truth_gpu[start:end, q_cosmo:].copy_(torch.from_numpy(tx_flat))
        del x, x_flat, tx, tx_flat
        print(f"  job {jid}: cosmo + field -> GPU ({time.time() - t0:.1f}s)")

    return posterior_gpu, truth_gpu, q_cosmo, q_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "samples_dir",
        nargs="?",
        default="tarp_results/conditional",
        help="Folder with x_samples_job_*.npy, true_x_job_*.npy, "
             "cosmo_samples_job_*.npy, true_cosmo_job_*.npy.",
    )
    parser.add_argument("--output", default="tarp_results/pokie_joint.pdf")
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--save-npz",
                        default="tarp_results/pokie_joint_scores.npz")
    parser.add_argument("--norm", action="store_true",
                        help="Apply scalar normalisation. Off by default "
                             "because the field swamps the global min/max "
                             "and reduces cosmology's contribution to "
                             "nearly zero after rescaling.")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Use only the first N job-chunks of samples "
                             "(default: all available).")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: running on CPU; this will be slow at joint q≈81926.")

    t0 = time.time()
    posterior_gpu, truth_gpu, q_cosmo, q_field = load_joint_to_gpu(
        args.samples_dir, device, max_jobs=args.max_jobs
    )
    print(f"load done in {time.time() - t0:.1f}s. "
          f"posterior {tuple(posterior_gpu.shape)}, truth {tuple(truth_gpu.shape)}")
    if device.type == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    print(f"normalisation: {'scalar (norm=True)' if args.norm else 'OFF'}")

    torch.manual_seed(2024)
    t0 = time.time()
    boot = pokie_bootstrap_efficient(
        truth_gpu, posterior_gpu,
        num_bootstrap=args.num_bootstrap,
        num_runs=args.num_runs, device=device,
        norm=args.norm,
    )
    t_compute = time.time() - t0
    boot_np = boot.detach().cpu().numpy()[:, 0]
    mean, std = boot_np.mean(), boot_np.std()
    print(f"pokie joint score: {mean:.6f} ± {std:.6f}   "
          f"(wall {t_compute:.1f}s, {args.num_runs} MC × "
          f"{args.num_bootstrap} bootstrap)")
    np.savez(args.save_npz, boot=boot_np, mean=mean, std=std,
             q_cosmo=q_cosmo, q_field=q_field, norm=args.norm)
    print(f"wrote {args.save_npz}")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.axvline(2 / 3, ls="--", color="k", label="Well-calibrated (2/3)")
    ax.axvline(1 / 2, ls=":", color="gray", label="Misaligned (1/2)")
    ax.hist(boot_np, bins=20, color="tab:blue", alpha=0.6,
            edgecolor="tab:blue", label="Bootstrap")
    ax.axvline(mean, color="tab:blue", lw=2, label=f"Mean = {mean:.3f}")
    for k in (1, 2, 3):
        ax.axvspan(mean - k * std, mean + k * std, color="tab:blue", alpha=0.08)
    ax.set_xlabel("Pokie score (joint κ, θ)")
    ax.set_ylabel("Count")
    ax.set_xlim(0.45, 0.75)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output)
    base, _ = os.path.splitext(args.output)
    fig.savefig(base + ".png", dpi=200)
    print(f"wrote {args.output} and {base}.png")


if __name__ == "__main__":
    main()
