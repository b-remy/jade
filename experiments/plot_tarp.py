"""TARP coverage of the posterior samples written by sample_calibration.py.

``--space cosmo`` covers the marginal p(theta | y) on CPU via
``tarp.get_tarp_coverage``; ``--space joint`` covers p(theta, kappa | y) with the
flattened field concatenated after the 6 cosmology dimensions, which needs
``tarp.get_tarp_coverage_efficient`` on a GPU (the tensor is ~82 GB).
"""

import argparse
import glob
import os
import re
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    }
)

from jade.paths import RESULTS_DIR


def _job_ids(samples_dir, space):
    stem = "cosmo_samples_job_" if space == "cosmo" else "x_samples_job_"
    files = sorted(
        glob.glob(os.path.join(samples_dir, stem + "*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No {stem}*.npy files found in {samples_dir}")

    needed = ["cosmo_samples_job_{}.npy", "true_cosmo_job_{}.npy"]
    if space == "joint":
        needed += ["x_samples_job_{}.npy", "true_x_job_{}.npy"]

    ids = []
    for f in files:
        jid = int(re.search(r"_(\d+)\.npy$", f).group(1))
        if all(os.path.exists(os.path.join(samples_dir, n.format(jid))) for n in needed):
            ids.append(jid)
        else:
            print(f"warning: skipping job {jid}, incomplete sample files")
    return ids


def load_cosmo(samples_dir):
    """Cosmology samples in the TARP convention: (n_samples, n_obs, q_cosmo)."""
    ids = _job_ids(samples_dir, "cosmo")
    samples, truths = [], []
    for jid in ids:
        samples.append(np.load(os.path.join(samples_dir, f"cosmo_samples_job_{jid}.npy")))
        truths.append(np.load(os.path.join(samples_dir, f"true_cosmo_job_{jid}.npy")))

    samples = np.concatenate(samples, axis=0)
    truths = np.concatenate(truths, axis=0)
    return np.transpose(samples, [1, 0, 2]), truths


def load_joint_to_gpu(samples_dir, device, max_samples=None, max_obs=None):
    """Stream-load per-job files into GPU tensors in the TARP convention:
    samples (n_samples, n_obs, q_total), truth (n_obs, q_total), with
    q_total = q_cosmo + q_field and cosmo in the first q_cosmo channels.
    """
    import torch

    ids = _job_ids(samples_dir, "joint")
    head = np.load(os.path.join(samples_dir, f"x_samples_job_{ids[0]}.npy"), mmap_mode="r")
    c0 = np.load(os.path.join(samples_dir, f"cosmo_samples_job_{ids[0]}.npy"), mmap_mode="r")
    obs_per_job, n_samples_full, *spatial = head.shape
    q_field = int(np.prod(spatial))
    q_cosmo = c0.shape[-1]
    q_total = q_cosmo + q_field
    n_obs_total = obs_per_job * len(ids)
    del head, c0
    n_samples = n_samples_full if max_samples is None else min(n_samples_full, max_samples)
    if max_obs is not None:
        n_obs_total = min(n_obs_total, max_obs)

    print(
        f"Allocating joint samples on {device}: (n_samples={n_samples}, "
        f"n_obs={n_obs_total}, q={q_total}) = "
        f"{n_samples * n_obs_total * q_total * 4 / 2**30:.1f} GiB "
        f"(q_cosmo={q_cosmo}, q_field={q_field})"
    )
    samples_gpu = torch.empty((n_samples, n_obs_total, q_total), dtype=torch.float32, device=device)
    truths_gpu = torch.empty((n_obs_total, q_total), dtype=torch.float32, device=device)

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
        samples_gpu[:, start:end, :q_cosmo].copy_(torch.from_numpy(c_swapped[:, :actual, :]))
        del c_swapped
        tc = np.load(os.path.join(samples_dir, f"true_cosmo_job_{jid}.npy"))
        truths_gpu[start:end, :q_cosmo].copy_(torch.from_numpy(tc[:actual]))
        del tc

        # field: (obs_per_job, n_samples, 128,128,5) -> (n_samples, obs_per_job, q_field)
        x = np.load(os.path.join(samples_dir, f"x_samples_job_{jid}.npy"))
        if max_samples is not None and x.shape[1] > n_samples:
            x = x[:, :n_samples]
        x_swapped = np.ascontiguousarray(x.reshape(obs_per_job, n_samples, q_field).transpose(1, 0, 2))
        del x
        samples_gpu[:, start:end, q_cosmo:].copy_(torch.from_numpy(x_swapped[:, :actual, :]))
        del x_swapped
        tx = np.load(os.path.join(samples_dir, f"true_x_job_{jid}.npy")).reshape(obs_per_job, q_field)
        truths_gpu[start:end, q_cosmo:].copy_(torch.from_numpy(tx[:actual]))
        del tx

        obs_written = end
        print(f"  job {jid}: cosmo + field -> GPU ({time.time() - t0:.1f}s)")

    return samples_gpu, truths_gpu, q_cosmo, q_field


def plot_coverage(ecp, alpha, label, color, output):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal case")
    mean = ecp.mean(axis=0)
    std = ecp.std(axis=0)
    ax.plot(alpha, mean, label=label, color=color)
    for k in (1, 2, 3):
        ax.fill_between(alpha, mean - k * std, mean + k * std, color=color, alpha=0.2)
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


def coverage_cosmo(args):
    from tarp import get_tarp_coverage

    samples, truths = load_cosmo(args.samples_dir)
    print(f"Loaded samples {samples.shape}, truths {truths.shape}")

    if args.seed is not None:
        # tarp draws the bootstrap resample from the global RNG before applying
        # its own seed, so both have to be pinned for a reproducible result.
        np.random.seed(args.seed)

    kwargs = dict(
        references=args.references,
        metric=args.metric,
        norm=args.norm,
        bootstrap=True,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
    )
    if args.num_alpha_bins is not None:
        kwargs["num_alpha_bins"] = args.num_alpha_bins

    return get_tarp_coverage(samples, truths, **kwargs)


def coverage_joint(args):
    import torch
    from tarp import get_tarp_coverage_efficient

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available; the joint tensor is ~82 GB")
    device = torch.device("cuda")

    t0 = time.time()
    samples_gpu, truths_gpu, q_cosmo, q_field = load_joint_to_gpu(
        args.samples_dir,
        device,
        max_samples=args.max_samples,
        max_obs=args.max_obs,
    )
    print(f"load done in {time.time() - t0:.1f}s. samples {tuple(samples_gpu.shape)}, truth {tuple(truths_gpu.shape)}")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    print(f"normalisation: norm={args.norm} scalar_norm={args.scalar_norm}")

    torch.manual_seed(2024)
    t0 = time.time()
    ecp, alpha = get_tarp_coverage_efficient(
        samples_gpu,
        truths_gpu,
        references=args.references,
        norm=args.norm,
        scalar_norm=args.scalar_norm,
        bootstrap=True,
        num_bootstrap=args.num_bootstrap,
        num_alpha_bins=args.num_alpha_bins,
    )
    torch.cuda.synchronize()
    print(
        f"compute done in {time.time() - t0:.1f}s ({args.num_bootstrap} bootstrap iterations on q={q_cosmo + q_field})"
    )

    return ecp.cpu().numpy(), alpha.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_dir", nargs="?", default=str(RESULTS_DIR / "conditional"))
    parser.add_argument("--space", choices=["cosmo", "joint"], default="cosmo")
    parser.add_argument("--output", default=None, help="Defaults to tarp_<space>.pdf in the results dir.")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--num-alpha-bins", type=int, default=None)
    parser.add_argument("--references", default="random")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="--space cosmo only: makes the coverage "
        "reproducible. Left unset (the default, matching "
        "the paper runs) the bootstrap is not "
        "reproducible.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Per-dim [0,1] rescale via truth min/max. "
        "Recommended for the joint so the 6 cosmo dims "
        "and 81920 field dims are comparable per-dim.",
    )
    # cosmo only
    parser.add_argument("--metric", default="euclidean", help="--space cosmo only.")
    # joint only
    parser.add_argument(
        "--scalar-norm",
        action="store_true",
        help="--space joint only: global [0,1] rescale instead of per-dim. Mutually exclusive with --norm.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="--space joint only.")
    parser.add_argument("--max-obs", type=int, default=None, help="--space joint only.")
    args = parser.parse_args()

    output = args.output or str(RESULTS_DIR / f"tarp_{args.space}.pdf")

    if args.space == "cosmo":
        ecp, alpha = coverage_cosmo(args)
    else:
        ecp, alpha = coverage_joint(args)

    plot_coverage(ecp, alpha, label="JADE", color="tab:blue", output=output)


if __name__ == "__main__":
    main()
