"""Per-scale z-score: where (in spatial scale) is the kappa posterior mis-dispersed?

For each obs, in Fourier space per channel:
  posterior power   P_post(k) = < |FFT(sample - post_mean)|^2 >_samples
  truth power       P_true(k) =   |FFT(truth  - post_mean)|^2
radially binned in |k|. Per-scale z(k) = sqrt( <P_true>/<P_post> ) over modes+obs.

z(k) ~ 1  : calibrated at that scale.
z(k) >> 1 : posterior UNDER-dispersed at that scale (truth has more power than the
            posterior allows).  low-k (large scale) => mode-locking to conditioning.
Per-pixel z~1 with a scale-dependent z(k) means over/under-dispersion that CANCELS
in the marginal but breaks the joint (explains calibrated marginals + bad TARP).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.unicode_minus": False})

D = "/work/hdd/benb/bremy/jade/tarp/million_sde_g1.0"
JOBS = [0, 1]          # 50 obs
N_SAMP = 200           # subsample posterior samples (memory)
N, NCH = 128, 5

# radial |k| ring index for a 128x128 grid
kf = np.fft.fftfreq(N) * N
kx, ky = np.meshgrid(kf, kf, indexing="ij")
kbin = np.round(np.sqrt(kx**2 + ky**2)).astype(int)        # 0..~90
nb = kbin.max() + 1
ring = kbin.ravel()
counts = np.bincount(ring, minlength=nb)

sumPpost = np.zeros((nb, NCH)); sumPtrue = np.zeros((nb, NCH))

for j in JOBS:
    xs = np.load(f"{D}/x_samples_job_{j}.npy", mmap_mode="r")
    tx = np.load(f"{D}/true_x_job_{j}.npy")
    for i in range(xs.shape[0]):
        s = np.asarray(xs[i, :N_SAMP], np.float32)          # (Ns,128,128,5)
        pm = s.mean(0)
        ds = s - pm[None]
        Fds = np.fft.fft2(ds, axes=(1, 2))
        Ppost = (np.abs(Fds) ** 2).mean(0)                  # (128,128,5)
        Fdt = np.fft.fft2(tx[i] - pm, axes=(0, 1))
        Ptrue = np.abs(Fdt) ** 2                            # (128,128,5)
        for c in range(NCH):
            sumPpost[:, c] += np.bincount(ring, weights=Ppost[..., c].ravel(), minlength=nb)
            sumPtrue[:, c] += np.bincount(ring, weights=Ptrue[..., c].ravel(), minlength=nb)

z = np.sqrt(sumPtrue / np.clip(sumPpost, 1e-30, None))     # (nb, NCH)
zk = np.arange(nb)

# pooled over channels (sum power across channels)
z_pool = np.sqrt(sumPtrue.sum(1) / np.clip(sumPpost.sum(1), 1e-30, None))

# report a few scales
def at(k): return z_pool[min(k, nb - 1)]
print(f"per-scale z (pooled): k=1:{at(1):.2f}  k=2:{at(2):.2f}  k=4:{at(4):.2f}  "
      f"k=8:{at(8):.2f}  k=16:{at(16):.2f}  k=32:{at(32):.2f}  k=60:{at(60):.2f}")
print("(z~1 calibrated; z>1 under-dispersed at that scale; low k = large scale)")

fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
for c in range(NCH):
    ax.plot(zk[1:], z[1:, c], lw=1, alpha=0.6, label=f"bin {c}")
ax.plot(zk[1:], z_pool[1:], "k-", lw=2.2, label="pooled")
ax.axhline(1.0, ls="--", color="0.4", label="calibrated (z=1)")
ax.set_xscale("log")
ax.set_xlabel("spatial frequency |k|  (low = large scale)")
ax.set_ylabel("per-scale z  =  sqrt(P_true / P_post)")
ax.set_title("Per-scale field calibration (million SDE g=1)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
out = "tarp_results/diag_per_scale_z.pdf"
fig.savefig(out); fig.savefig(out[:-4] + ".png", dpi=200)
print("Wrote", out)
