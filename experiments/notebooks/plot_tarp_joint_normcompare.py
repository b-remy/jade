"""Joint (kappa, theta) TARP coverage in two panels: norm=True vs norm=False.

Loads the joint samples to GPU once (via plot_tarp_joint.load_joint_to_gpu) and
runs tarp.get_tarp_coverage_efficient twice -- per-dim normalised and raw -- so
the two normalisations can be compared side by side on the same samples.
"""

import argparse
import os
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

# Import-safe: plot_tarp_joint guards its CLI behind __main__.
from plot_tarp_joint import load_joint_to_gpu


def panel(ax, ecp, alpha, title, color="tab:blue"):
    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal case")
    mean = ecp.mean(axis=0)
    std = ecp.std(axis=0)
    ax.plot(alpha, mean, label="JADE", color=color)
    for k in (1, 2, 3):
        ax.fill_between(alpha, mean - k * std, mean + k * std,
                        color=color, alpha=0.2)
    ax.set_xlabel("Credibility level")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_dir", nargs="?",
                        default="tarp_results/million_sde_g1.0")
    parser.add_argument("--output",
                        default="tarp_results/tarp_joint_million_sde_g1_normcompare.pdf")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--references", default="random")
    args = parser.parse_args()

    import torch
    from tarp import get_tarp_coverage_efficient
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available; the joint tensor is ~76 GB")
    device = torch.device("cuda")

    t0 = time.time()
    samples_gpu, truths_gpu, q_cosmo, q_field = load_joint_to_gpu(
        args.samples_dir, device)
    print(f"load done in {time.time() - t0:.1f}s. "
          f"samples {tuple(samples_gpu.shape)}, truth {tuple(truths_gpu.shape)}")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    results = {}
    for norm in (True, False):
        torch.manual_seed(2024)  # same seed -> norm=True panel matches the single plot
        t0 = time.time()
        ecp, alpha = get_tarp_coverage_efficient(
            samples_gpu, truths_gpu,
            references=args.references,
            norm=norm,
            scalar_norm=False,
            bootstrap=True,
            num_bootstrap=args.num_bootstrap,
            num_alpha_bins=None,
        )
        torch.cuda.synchronize()
        print(f"norm={norm}: compute done in {time.time() - t0:.1f}s")
        results[norm] = (ecp.cpu().numpy(), alpha.cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    panel(axes[0], *results[True], title="norm=True (per-dim)")
    panel(axes[1], *results[False], title="norm=False (raw)")
    axes[0].set_ylabel("Expected coverage probability")
    fig.suptitle(f"Joint (kappa, theta) TARP -- {os.path.basename(args.samples_dir)}")
    fig.tight_layout()

    fig.savefig(args.output)
    base, _ = os.path.splitext(args.output)
    fig.savefig(base + ".png", dpi=200)
    print(f"Wrote {args.output} and {base}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
