"""Joint TARP with BLOCK-WISE scalar norm (separate scalar for theta vs kappa),
checked across several seeds for stability.

The built-in scalar_norm uses ONE global scalar over all 81926 dims, which is
dominated by the cosmo (theta) range and compresses kappa into a narrow band ->
seed-unstable. Here we rescale the 6 cosmo dims by the cosmo block's own
min/max and the 81920 field dims by the field block's own min/max, so BOTH land
on [0,1] (matching the estimator's U[0,1) references) while each block keeps its
internal geometry. Then call get_tarp_coverage_efficient with no further norm.
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


def block_scalar_norm_(samples, truths, q_cosmo):
    """In-place: rescale cosmo block and field block to [0,1] by their OWN
    (theta-based) global min/max, separately."""
    for lo, hi in [(0, q_cosmo), (q_cosmo, truths.shape[1])]:
        t_blk = truths[:, lo:hi]
        low = t_blk.amin()
        scale = (t_blk.amax() - low) + 1e-10
        truths[:, lo:hi].sub_(low).div_(scale)
        samples[:, :, lo:hi].sub_(low).div_(scale)
        print(f"  block [{lo}:{hi}]: low={low.item():+.4f} scale={scale.item():.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples_dir", nargs="?", default="tarp_results/million_sde_g1.0")
    p.add_argument("--output",
                   default="tarp_results/tarp_joint_million_sde_g1_blockscalar_seeds.pdf")
    p.add_argument("--num-bootstrap", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--max-obs", type=int, default=None,
                   help="Use only the first N observations (job order).")
    args = p.parse_args()

    import torch
    from tarp import get_tarp_coverage_efficient
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available; the joint tensor is ~76 GB")
    device = torch.device("cuda")

    t0 = time.time()
    samples_gpu, truths_gpu, q_cosmo, q_field = load_joint_to_gpu(
        args.samples_dir, device, max_obs=args.max_obs)
    print(f"load done in {time.time()-t0:.1f}s. samples {tuple(samples_gpu.shape)}")

    print("block-wise scalar norm:")
    block_scalar_norm_(samples_gpu, truths_gpu, q_cosmo)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal case")
    cmap = plt.get_cmap("viridis")
    rows = []
    for i, s in enumerate(args.seeds):
        ecp, alpha = get_tarp_coverage_efficient(
            samples_gpu, truths_gpu, references="random",
            norm=False, scalar_norm=False, bootstrap=True,
            num_bootstrap=args.num_bootstrap, seed=s)
        mean = ecp.mean(0).cpu().numpy()
        a = alpha.cpu().numpy()
        ax.plot(a, mean, color=cmap(i / max(1, len(args.seeds) - 1)),
                lw=1.4, label=f"seed {s}")
        pts = [np.interp(x, a, mean) for x in (0.2, 0.5, 0.8)]
        rows.append(pts)
        print(f"seed {s}: ECP@[0.2,0.5,0.8] = [{pts[0]:.3f}, {pts[1]:.3f}, {pts[2]:.3f}]")

    arr = np.array(rows)
    print(f"\nacross seeds  mean: {arr.mean(0).round(3)}  std: {arr.std(0).round(4)}")

    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Expected coverage probability")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Joint block-wise scalar-norm TARP -- seed stability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output)
    fig.savefig(os.path.splitext(args.output)[0] + ".png", dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
