"""Scalar-norm joint TARP recomputed under several seeds, overlaid.

Loads the joint once (GPU) and runs get_tarp_coverage_efficient(scalar_norm=True)
for several torch seeds, to show the run-to-run stochasticity (random references
+ bootstrap) of the coverage curve.
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
    "mathtext.fontset": "cm", "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
})

from plot_tarp_joint import load_joint_to_gpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples_dir", nargs="?", default="tarp_results/million_sde_g1.0")
    p.add_argument("--output",
                   default="tarp_results/tarp_joint_million_sde_g1_scalarnorm_seeds.pdf")
    p.add_argument("--num-bootstrap", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = p.parse_args()

    import torch
    from tarp import get_tarp_coverage_efficient
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available; the joint tensor is ~76 GB")
    device = torch.device("cuda")

    t0 = time.time()
    samples_gpu, truths_gpu, q_cosmo, q_field = load_joint_to_gpu(args.samples_dir, device)
    print(f"load done in {time.time()-t0:.1f}s. samples {tuple(samples_gpu.shape)}")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal case")
    cmap = plt.get_cmap("viridis")
    rows = []
    for i, s in enumerate(args.seeds):
        ecp, alpha = get_tarp_coverage_efficient(
            samples_gpu, truths_gpu, references="random",
            norm=False, scalar_norm=True, bootstrap=True,
            num_bootstrap=args.num_bootstrap, seed=s)
        mean = ecp.mean(0).cpu().numpy()
        a = alpha.cpu().numpy()
        ax.plot(a, mean, color=cmap(i / max(1, len(args.seeds) - 1)),
                lw=1.4, label=f"seed {s}")
        pts = [np.interp(x, a, mean) for x in (0.2, 0.5, 0.8)]
        rows.append((s, *pts))
        print(f"seed {s}: ECP@[0.2,0.5,0.8] = [{pts[0]:.3f}, {pts[1]:.3f}, {pts[2]:.3f}]")

    arr = np.array([r[1:] for r in rows])
    print(f"\nacross seeds  mean: {arr.mean(0).round(3)}  std: {arr.std(0).round(4)}")

    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Expected coverage probability")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Joint scalar-norm TARP -- seed stochasticity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output)
    fig.savefig(os.path.splitext(args.output)[0] + ".png", dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
