import argparse
import glob
import os
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# Paper-style typography without a system TeX install. ``mathtext.fontset='cm'``
# uses Computer Modern (bundled with matplotlib) for everything inside $...$,
# while plain text falls back to whatever serif font is available (DejaVu Serif
# on most clusters). Math labels wrapped in $...$ end up looking close to
# pdflatex output.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

from tarp import get_tarp_coverage


def load_samples(samples_dir):
    sample_files = sorted(
        glob.glob(os.path.join(samples_dir, "cosmo_samples_job_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    if not sample_files:
        raise FileNotFoundError(
            f"No cosmo_samples_job_*.npy files found in {samples_dir}"
        )

    samples, truths = [], []
    for sf in sample_files:
        job_id = re.search(r"_(\d+)\.npy$", sf).group(1)
        tf = os.path.join(samples_dir, f"true_cosmo_job_{job_id}.npy")
        if not os.path.exists(tf):
            print(f"warning: skipping job {job_id}, missing {tf}")
            continue
        samples.append(np.load(sf))
        truths.append(np.load(tf))

    samples = np.concatenate(samples, axis=0)
    truths = np.concatenate(truths, axis=0)
    return samples, truths


def main():
    parser = argparse.ArgumentParser(
        description="Compute the cosmology-only TARP coverage plot from saved "
                    "posterior/prior samples (as written by tarp_conditional.py)."
    )
    parser.add_argument(
        "samples_dir",
        nargs="?",
        default="tarp_results/conditional",
        help="Folder containing cosmo_samples_job_*.npy and true_cosmo_job_*.npy",
    )
    parser.add_argument("--output", default="tarp_results/tarp_conditional.pdf",
                        help="Output figure path (pdf/png).")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--num-alpha-bins", type=int, default=None,
                        help="If set, passed through to get_tarp_coverage.")
    parser.add_argument("--norm", action="store_true",
                        help="Pass norm=True to get_tarp_coverage (per-dim normalization).")
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--references", default="random")
    args = parser.parse_args()

    samples, truths = load_samples(args.samples_dir)
    print(f"Loaded samples {samples.shape}, truths {truths.shape}")

    # tarp expects samples shape (n_samples, n_sims, n_dims)
    samples_tarp = np.transpose(samples, [1, 0, 2])
    truths_tarp = truths

    coverage_kwargs = dict(
        references=args.references,
        metric=args.metric,
        norm=args.norm,
        bootstrap=True,
        num_bootstrap=args.num_bootstrap,
        
    )
    if args.num_alpha_bins is not None:
        coverage_kwargs["num_alpha_bins"] = args.num_alpha_bins

    ecp, alpha = get_tarp_coverage(samples_tarp, truths_tarp, **coverage_kwargs)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal case")
    mean = ecp.mean(axis=0)
    std = ecp.std(axis=0)
    ax.plot(alpha, mean, label="JADE", color="tab:blue")
    for k in (1, 2, 3):
        ax.fill_between(alpha, mean - k * std, mean + k * std,
                        color="tab:blue", alpha=0.2)
    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Expected coverage probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Wrote {args.output}")
    base, _ = os.path.splitext(args.output)
    png_output = base + ".png"
    fig.savefig(png_output, dpi=200)
    print(f"Wrote {png_output}")


if __name__ == "__main__":
    main()
