"""Joint (κ, θ) TARP coverage on the posterior samples written by
tarp_conditional.py — TARP analog of plot_mira_joint.py.

Concatenates the 6-d cosmology vector with the flattened (128, 128, 5) =
81920-d field into a single q = 81926 vector per sample, then runs
``tarp.get_tarp_coverage_efficient`` on the GPU (the 82 GB samples tensor
is the same size as the field-only case, so the GPU path is required).

Layout matches plot_mira_joint.py: cosmo occupies the first q_cosmo
channels, field the remaining q_field.
"""

import argparse
import glob
import os
import re
import time

import numpy as np
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


def _job_ids(samples_dir):
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
    return ids


def load_joint_to_gpu(samples_dir, device, max_samples=None, max_obs=None):
    """Stream-load per-job files into GPU tensors in the TARP convention:
    samples (n_samples, n_obs, q_total), truth (n_obs, q_total), with
    q_total = q_cosmo + q_field and cosmo in the first q_cosmo channels.
    """
    import torch
    ids = _job_ids(samples_dir)
    head = np.load(os.path.join(samples_dir, f"x_samples_job_{ids[0]}.npy"),
                   mmap_mode="r")
    c0 = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{ids[0]}.npy"),
                 mmap_mode="r")
    obs_per_job, n_samples_full, *spatial = head.shape
    q_field = int(np.prod(spatial))
    q_cosmo = c0.shape[-1]
    q_total = q_cosmo + q_field
    n_obs_total = obs_per_job * len(ids)
    del head, c0
    n_samples = n_samples_full if max_samples is None else min(n_samples_full, max_samples)
    if max_obs is not None:
        n_obs_total = min(n_obs_total, max_obs)

    print(f"Allocating joint samples on {device}: (n_samples={n_samples}, "
          f"n_obs={n_obs_total}, q={q_total}) = "
          f"{n_samples * n_obs_total * q_total * 4 / 2**30:.1f} GiB "
          f"(q_cosmo={q_cosmo}, q_field={q_field})")
    samples_gpu = torch.empty((n_samples, n_obs_total, q_total),
                              dtype=torch.float32, device=device)
    truths_gpu = torch.empty((n_obs_total, q_total),
                             dtype=torch.float32, device=device)

    obs_written = 0
    for jid in ids:
        if obs_written >= n_obs_total:
            break
        t0 = time.time()
        start = obs_written
        actual = min(obs_per_job, n_obs_total - start)
        end = start + actual

        # cosmo: (obs_per_job, n_samples, q_cosmo) -> (n_samples, obs_per_job, q_cosmo)
        c = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{jid}.npy"))
        if max_samples is not None and c.shape[1] > n_samples:
            c = c[:, :n_samples]
        c_swapped = np.ascontiguousarray(c.transpose(1, 0, 2))
        del c
        samples_gpu[:, start:end, :q_cosmo].copy_(
            torch.from_numpy(c_swapped[:, :actual, :]))
        del c_swapped
        tc = np.load(os.path.join(samples_dir, f"true_cosmo_job_{jid}.npy"))
        truths_gpu[start:end, :q_cosmo].copy_(torch.from_numpy(tc[:actual]))
        del tc

        # field: (obs_per_job, n_samples, 128,128,5) -> (n_samples, obs_per_job, q_field)
        x = np.load(os.path.join(samples_dir, f"x_samples_job_{jid}.npy"))
        if max_samples is not None and x.shape[1] > n_samples:
            x = x[:, :n_samples]
        x_swapped = np.ascontiguousarray(
            x.reshape(obs_per_job, n_samples, q_field).transpose(1, 0, 2)
        )
        del x
        samples_gpu[:, start:end, q_cosmo:].copy_(
            torch.from_numpy(x_swapped[:, :actual, :]))
        del x_swapped
        tx = np.load(os.path.join(samples_dir, f"true_x_job_{jid}.npy")
                     ).reshape(obs_per_job, q_field)
        truths_gpu[start:end, q_cosmo:].copy_(torch.from_numpy(tx[:actual]))
        del tx

        obs_written = end
        print(f"  job {jid}: cosmo + field -> GPU ({time.time() - t0:.1f}s)")

    return samples_gpu, truths_gpu, q_cosmo, q_field


def plot_one(ecp, alpha, label, color, output):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal case")
    mean = ecp.mean(axis=0)
    std = ecp.std(axis=0)
    ax.plot(alpha, mean, label=label, color=color)
    for k in (1, 2, 3):
        ax.fill_between(alpha, mean - k * std, mean + k * std,
                        color=color, alpha=0.2)
    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Expected coverage probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}")
    base, _ = os.path.splitext(output)
    png = base + ".png"
    fig.savefig(png, dpi=200)
    print(f"Wrote {png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_dir", nargs="?",
                        default="tarp_results/conditional")
    parser.add_argument("--output", default="tarp_results/tarp_joint.pdf")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--num-alpha-bins", type=int, default=None)
    parser.add_argument("--references", default="random")
    parser.add_argument("--norm", action="store_true",
                        help="Per-dim [0,1] rescale via truth min/max "
                             "(tarp's norm). Recommended for the joint so "
                             "the 6 cosmo dims and 81920 field dims are "
                             "comparable per-dim.")
    parser.add_argument("--scalar-norm", action="store_true",
                        help="Global [0,1] rescale instead of per-dim. "
                             "Mutually exclusive with --norm.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-obs", type=int, default=None)
    args = parser.parse_args()

    import torch
    from tarp import get_tarp_coverage_efficient
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available; the joint tensor is ~82 GB")
    device = torch.device("cuda")

    t0 = time.time()
    samples_gpu, truths_gpu, q_cosmo, q_field = load_joint_to_gpu(
        args.samples_dir, device,
        max_samples=args.max_samples, max_obs=args.max_obs,
    )
    print(f"load done in {time.time() - t0:.1f}s. "
          f"samples {tuple(samples_gpu.shape)}, truth {tuple(truths_gpu.shape)}")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    print(f"normalisation: norm={args.norm} scalar_norm={args.scalar_norm}")

    torch.manual_seed(2024)
    t0 = time.time()
    ecp, alpha = get_tarp_coverage_efficient(
        samples_gpu, truths_gpu,
        references=args.references,
        norm=args.norm,
        scalar_norm=args.scalar_norm,
        bootstrap=True,
        num_bootstrap=args.num_bootstrap,
        num_alpha_bins=args.num_alpha_bins,
    )
    torch.cuda.synchronize()
    print(f"compute done in {time.time() - t0:.1f}s "
          f"({args.num_bootstrap} bootstrap iterations on q={q_cosmo + q_field})")

    ecp = ecp.cpu().numpy()
    alpha = alpha.cpu().numpy()
    plot_one(ecp, alpha, label="JADE", color="tab:blue", output=args.output)


if __name__ == "__main__":
    main()
