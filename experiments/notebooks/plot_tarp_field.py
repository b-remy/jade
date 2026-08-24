"""TARP coverage on the κ-field posterior samples (analog of plot_tarp_conditional.py).

Reads x_samples_job_*.npy / true_x_job_*.npy from tarp_conditional.py. By
default flattens the (128, 128, 5) field per sample into a single 81920-d
vector and produces one coverage curve, directly analogous to the 6-d
cosmology TARP. ``--per-bin`` does five lighter per-redshift-bin curves
instead.
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

from tarp import get_tarp_coverage


def load_samples(samples_dir, max_samples=None, max_obs=None):
    sample_files = sorted(
        glob.glob(os.path.join(samples_dir, "x_samples_job_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    if not sample_files:
        raise FileNotFoundError(
            f"No x_samples_job_*.npy files found in {samples_dir}"
        )

    samples, truths = [], []
    for sf in sample_files:
        job_id = re.search(r"_(\d+)\.npy$", sf).group(1)
        tf = os.path.join(samples_dir, f"true_x_job_{job_id}.npy")
        if not os.path.exists(tf):
            print(f"warning: skipping job {job_id}, missing {tf}")
            continue
        x = np.load(sf)            # (n_obs, n_samples, 128, 128, 5)
        t = np.load(tf)            # (n_obs, 128, 128, 5)
        if max_samples is not None and x.shape[1] > max_samples:
            x = x[:, :max_samples]
        samples.append(x)
        truths.append(t)

    samples = np.concatenate(samples, axis=0)
    truths = np.concatenate(truths, axis=0)
    if max_obs is not None and samples.shape[0] > max_obs:
        samples = samples[:max_obs]
        truths = truths[:max_obs]
    return samples, truths


def run_tarp(samples_flat, truths_flat, args):
    """samples_flat: (n_obs, n_samples, n_dims); truths_flat: (n_obs, n_dims)."""
    samples_tarp = samples_flat.transpose(1, 0, 2)  # (n_samples, n_obs, n_dims)
    kwargs = dict(
        references=args.references,
        metric=args.metric,
        norm=args.norm,
        bootstrap=True,
        num_bootstrap=args.num_bootstrap,
    )
    if args.num_alpha_bins is not None:
        kwargs["num_alpha_bins"] = args.num_alpha_bins
    return get_tarp_coverage(samples_tarp, truths_flat, **kwargs)


def load_to_gpu_tarp(samples_dir, device, max_samples=None, max_obs=None):
    """Stream-load per-job files into pre-allocated GPU tensors in the
    TARP convention: samples (n_samples, n_obs, q), truth (n_obs, q).

    Host RAM peak: ~32 GB per file (the on-disk arr + a transposed copy);
    GPU peak: sizeof(samples).
    """
    import torch
    files = sorted(
        glob.glob(os.path.join(samples_dir, "x_samples_job_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    head = np.load(files[0], mmap_mode="r")
    obs_per_job, n_samples_full, *spatial = head.shape
    q = int(np.prod(spatial))
    n_jobs = len(files)
    n_obs_total = obs_per_job * n_jobs
    del head
    n_samples = n_samples_full if max_samples is None else min(n_samples_full, max_samples)
    if max_obs is not None:
        n_obs_total = min(n_obs_total, max_obs)

    print(f"Allocating samples on {device}: (n_samples={n_samples}, "
          f"n_obs={n_obs_total}, q={q}) = "
          f"{n_samples * n_obs_total * q * 4 / 2**30:.1f} GiB")
    samples_gpu = torch.empty((n_samples, n_obs_total, q),
                              dtype=torch.float32, device=device)
    truths_gpu = torch.empty((n_obs_total, q),
                             dtype=torch.float32, device=device)

    obs_written = 0
    for xf in files:
        if obs_written >= n_obs_total:
            break
        jid = int(re.search(r"_(\d+)\.npy$", xf).group(1))
        tf = os.path.join(samples_dir, f"true_x_job_{jid}.npy")
        if not os.path.exists(tf):
            print(f"warning: skipping job {jid}, missing {tf}")
            continue

        t0 = time.time()
        x = np.load(xf)  # (obs_per_job, n_samples_full, 128, 128, 5)
        if max_samples is not None and x.shape[1] > n_samples:
            x = x[:, :n_samples]
        # (obs_per_job, n_samples, q) → (n_samples, obs_per_job, q)
        x_swapped = np.ascontiguousarray(
            x.reshape(obs_per_job, n_samples, q).transpose(1, 0, 2)
        )
        del x

        start = obs_written
        actual = min(obs_per_job, n_obs_total - start)
        end = start + actual
        samples_gpu[:, start:end, :].copy_(torch.from_numpy(x_swapped[:, :actual, :]))
        del x_swapped

        t = np.load(tf).reshape(obs_per_job, q)
        truths_gpu[start:end].copy_(torch.from_numpy(t[:actual]))
        del t

        obs_written = end
        print(f"  job {jid}: -> GPU ({time.time() - t0:.1f}s)")

    return samples_gpu, truths_gpu


def run_tarp_gpu(samples_dir, args):
    """GPU path: stream-load to device, call get_tarp_coverage_efficient."""
    import torch
    from tarp import get_tarp_coverage_efficient
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu requested but no CUDA device available")
    device = torch.device("cuda")

    t0 = time.time()
    samples_gpu, truths_gpu = load_to_gpu_tarp(
        samples_dir, device,
        max_samples=args.max_samples, max_obs=args.max_obs,
    )
    t_load = time.time() - t0
    print(f"load done in {t_load:.1f}s. "
          f"samples {tuple(samples_gpu.shape)}, truth {tuple(truths_gpu.shape)}")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

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
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_compute = time.time() - t0
    print(f"compute done in {t_compute:.1f}s "
          f"({args.num_bootstrap} bootstrap iterations on q={samples_gpu.shape[-1]})")
    return ecp.cpu().numpy(), alpha.cpu().numpy()


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


def plot_per_bin(ecps, alpha, output):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal case")
    cmap = plt.get_cmap("viridis")
    for b, ecp in enumerate(ecps):
        mean = ecp.mean(axis=0)
        std = ecp.std(axis=0)
        c = cmap(b / max(len(ecps) - 1, 1))
        ax.plot(alpha, mean, color=c, label=f"Bin {b}")
        ax.fill_between(alpha, mean - std, mean + std, color=c, alpha=0.2)
    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Expected coverage probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}")
    base, _ = os.path.splitext(output)
    png = base + ".png"
    fig.savefig(png, dpi=200)
    print(f"Wrote {png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="TARP on the κ-field posterior samples written by tarp_conditional.py."
    )
    parser.add_argument(
        "samples_dir",
        nargs="?",
        default="tarp_results/conditional",
        help="Folder containing x_samples_job_*.npy and true_x_job_*.npy",
    )
    parser.add_argument("--output", default="tarp_results/tarp_field.pdf",
                        help="Output figure path (pdf/png).")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--num-alpha-bins", type=int, default=None)
    parser.add_argument("--norm", action="store_true",
                        help="Pass norm=True to get_tarp_coverage.")
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--references", default="random")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample posterior draws per observation.")
    parser.add_argument("--max-obs", type=int, default=None,
                        help="Subsample observations.")
    parser.add_argument("--per-bin", action="store_true",
                        help="One TARP curve per redshift bin (5 lighter curves) "
                             "instead of one curve on the flattened field.")
    parser.add_argument("--gpu", action="store_true",
                        help="Use tarp.get_tarp_coverage_efficient on GPU "
                             "(stream-loads to device, avoids the 82 GB CPU "
                             "broadcast intermediate). Flattened mode only.")
    parser.add_argument("--scalar-norm", action="store_true",
                        help="GPU path only. Scalar (global) [0, 1] "
                             "rescale via truth.min/max instead of tarp's "
                             "per-dim norm — preserves spatial structure "
                             "for field samples. Mutually exclusive with "
                             "--norm.")
    args = parser.parse_args()

    if args.gpu:
        if args.per_bin:
            raise SystemExit("--gpu and --per-bin are not currently combined")
        ecp, alpha = run_tarp_gpu(args.samples_dir, args)
        plot_one(ecp, alpha, label="JADE", color="tab:blue", output=args.output)
        return

    samples, truths = load_samples(args.samples_dir,
                                   max_samples=args.max_samples,
                                   max_obs=args.max_obs)
    print(f"Loaded samples {samples.shape}, truths {truths.shape}")

    n_obs, n_samples = samples.shape[:2]

    if args.per_bin:
        ecps = []
        alpha = None
        for b in range(samples.shape[-1]):
            s_b = samples[..., b].reshape(n_obs, n_samples, -1)
            t_b = truths[..., b].reshape(n_obs, -1)
            print(f"bin {b}: samples {s_b.shape}, truths {t_b.shape}")
            ecp, alpha = run_tarp(s_b, t_b, args)
            ecps.append(ecp)
        plot_per_bin(ecps, alpha, args.output)
    else:
        n_dims = int(np.prod(samples.shape[2:]))
        samples_flat = samples.reshape(n_obs, n_samples, n_dims)
        truths_flat = truths.reshape(n_obs, n_dims)
        print(f"TARP arrays: samples {samples_flat.shape}, truths {truths_flat.shape}")
        ecp, alpha = run_tarp(samples_flat, truths_flat, args)
        plot_one(ecp, alpha, label="JADE", color="tab:blue", output=args.output)


if __name__ == "__main__":
    main()
