"""Audit the SDE posterior samples vs the (known-good) ODE samples + truth.

If the SDE field/cosmo samples match the ODE ones in scale and have no NaN/inf,
the samples are fine and the norm=False joint degeneracy is the cosmo/field
scale-mixing artifact (not a sampling bug). If they are off-scale, that's a bug.
"""
import numpy as np

SDE = "/work/hdd/benb/bremy/jade/tarp/million_sde_g1.0"
ODE = "/work/hdd/benb/bremy/jade/tarp/million"   # 500-obs Heun ODE run


def stats(a):
    a = np.asarray(a, np.float64)
    return dict(mean=a.mean(), std=a.std(), min=a.min(), max=a.max(),
                nan=int(np.isnan(a).sum()), inf=int(np.isinf(a).sum()))


def line(tag, s):
    print(f"  {tag:14s} mean={s['mean']:+.4f} std={s['std']:.4f} "
          f"min={s['min']:+.3f} max={s['max']:+.3f} nan={s['nan']} inf={s['inf']}")


for name, jid in [("SDE", 0), ("ODE", 0)]:
    d = SDE if name == "SDE" else ODE
    xf = f"{d}/x_samples_job_{jid}.npy"
    x = np.load(xf, mmap_mode="r")[:5]      # 5 obs, all 500 samples
    cs = np.load(f"{d}/cosmo_samples_job_{jid}.npy")
    tx = np.load(f"{d}/true_x_job_{jid}.npy")
    tc = np.load(f"{d}/true_cosmo_job_{jid}.npy")
    print(f"\n=== {name}  (x_samples {np.load(xf, mmap_mode='r').shape}) ===")
    line("field samp", stats(x))
    line("field truth", stats(tx))
    line("cosmo samp", stats(cs))
    line("cosmo truth", stats(tc))

# Why norm=False joint can degenerate: compare the squared-distance budget of the
# 6 cosmo dims vs the 81920 field dims (raw, no per-dim norm), using SDE truths.
tc = np.load(f"{SDE}/true_cosmo_job_0.npy").astype(np.float64)      # (25,6)
tx = np.load(f"{SDE}/true_x_job_0.npy").astype(np.float64).reshape(25, -1)  # (25,81920)
cs = np.load(f"{SDE}/cosmo_samples_job_0.npy").astype(np.float64)   # (25,500,6)
xs = np.load(f"{SDE}/x_samples_job_0.npy").astype(np.float64).reshape(25, 500, -1)
# mean squared per-obs distance truth<->posterior-mean, summed over each block
dc = ((tc - cs.mean(1)) ** 2).sum(1).mean()
dx = ((tx - xs.mean(1)) ** 2).sum(1).mean()
print(f"\nnorm=False joint distance budget (truth vs post-mean, summed over block):")
print(f"  cosmo block (6 dims):     {dc:.4e}")
print(f"  field block (81920 dims): {dx:.4e}")
print(f"  ratio field/cosmo = {dx/dc:.1f}  (if >>1 or <<1, raw joint is dominated "
      f"by one block -> norm=False joint is meaningless by construction)")
