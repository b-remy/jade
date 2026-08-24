"""One inflation factor c per process (avoids holding two 76GB tensors).

Rescales FIELD posterior samples about their per-obs mean: field'=mean+c*(field-mean),
block-scalar-norms, runs joint TARP over seeds, saves (alpha, ecp_mean) to npz.
"""
import argparse, os
import numpy as np

from plot_tarp_joint import load_joint_to_gpu
from plot_tarp_joint_blockscalar_seeds import block_scalar_norm_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples_dir", nargs="?", default="tarp_results/million_sde_g1.0")
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--num-bootstrap", type=int, default=100)
    p.add_argument("--out-npz", required=True)
    args = p.parse_args()

    import torch
    from tarp import get_tarp_coverage_efficient
    device = torch.device("cuda")

    samples, truths, q_cosmo, q_field = load_joint_to_gpu(args.samples_dir, device)
    fld = samples[:, :, q_cosmo:]
    m = fld.mean(dim=0, keepdim=True)          # (1,T,qf), small
    # field' = c*field + (1-c)*m, fully in place (no 76GB temporaries)
    fld.mul_(args.c)
    fld.add_(m, alpha=(1.0 - args.c))
    block_scalar_norm_(samples, truths, q_cosmo)

    ecps = []
    for s in args.seeds:
        ecp, alpha = get_tarp_coverage_efficient(
            samples, truths, references="random", norm=False, scalar_norm=False,
            bootstrap=True, num_bootstrap=args.num_bootstrap, seed=s)
        ecps.append(ecp.mean(0).cpu().numpy())
    a = alpha.cpu().numpy()
    ecp_mean = np.mean(ecps, 0)
    pts = [np.interp(x, a, ecp_mean) for x in (0.2, 0.5, 0.8)]
    print(f"c={args.c:.3f}: ECP@[0.2,0.5,0.8]=[{pts[0]:.3f},{pts[1]:.3f},{pts[2]:.3f}]", flush=True)
    np.savez(args.out_npz, alpha=a, ecp_mean=ecp_mean, c=args.c)
    print("Wrote", args.out_npz)


if __name__ == "__main__":
    main()
