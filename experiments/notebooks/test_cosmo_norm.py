"""Cosmo-only TARP (6 dims) with norm=False vs norm=True, on CPU.

Isolates whether the raw cosmo scale (values outside [0,1], vs the estimator's
U[0,1) references) is what degenerates the norm=False joint TARP.
"""
import glob, re
import numpy as np
import torch
from tarp import get_tarp_coverage_efficient

D = "/work/hdd/benb/bremy/jade/tarp/million_sde_g1.0"
ids = sorted(int(re.search(r"_(\d+)\.npy$", p).group(1))
             for p in glob.glob(f"{D}/cosmo_samples_job_*.npy"))

cs = np.concatenate([np.load(f"{D}/cosmo_samples_job_{j}.npy") for j in ids])  # (500,500,6)
tc = np.concatenate([np.load(f"{D}/true_cosmo_job_{j}.npy") for j in ids])     # (500,6)
samples = torch.from_numpy(np.ascontiguousarray(cs.transpose(1, 0, 2))).float()  # (500,500,6)
truth = torch.from_numpy(tc).float()
print("cosmo raw per-dim range:")
for d in range(6):
    print(f"  dim{d}: [{tc[:,d].min():+.3f}, {tc[:,d].max():+.3f}]  "
          f"(U[0,1) refs sit at ~0.5)")

for norm in (False, True):
    torch.manual_seed(2024)
    ecp, alpha = get_tarp_coverage_efficient(
        samples.clone(), truth.clone(), references="random",
        norm=norm, bootstrap=True, num_bootstrap=100)
    m = ecp.mean(0).numpy(); a = alpha.numpy()
    # report ECP at a few credibility levels
    pts = [np.interp(x, a, m) for x in (0.2, 0.5, 0.8)]
    print(f"norm={norm!s:5s}  ECP@[0.2,0.5,0.8] = "
          f"[{pts[0]:.2f}, {pts[1]:.2f}, {pts[2]:.2f}]  (ideal 0.2/0.5/0.8)")
