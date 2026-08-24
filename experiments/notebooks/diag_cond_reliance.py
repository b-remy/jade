"""Conditioning-over-reliance check: is the kappa posterior narrower than the
observational noise allows, and is the truth many posterior-sigmas from the mean?

For each obs we have 500 posterior kappa samples (raw units) and the true kappa.
Per channel we report:
  - mean posterior per-pixel std   s_model
  - sigma_lsst                     the noise added to build the conditioning y
  - sigma_post (Gaussian ref)      sigma_lsst*FIELD_STD/sqrt(sigma_lsst^2+FIELD_STD^2)
                                   (analytic per-pixel posterior std, flat-ish prior)
  - field z-score                  z=(kappa_true-post_mean)/post_std
      RMS(z), E|z|, frac(|z|<1)    calibrated Gaussian => RMS~1, E|z|~0.8, frac~0.68
RMS(z) >> 1  AND  s_model << sigma_post  =>  overconfident / conditioning over-reliance.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from jade.init import FIELD_STD, sigma_lsst

D = "/work/hdd/benb/bremy/jade/tarp/million_sde_g1.0"
JOBS = [0, 1, 2, 3]          # first 100 obs (plenty for per-channel stats)
NCH = 5

fstd = np.asarray(FIELD_STD).reshape(-1)
slsst = np.asarray(sigma_lsst).reshape(-1)
print("FIELD_STD per ch:", np.round(fstd, 4))
print("sigma_lsst per ch:", np.round(slsst, 4))

# accumulators per channel
sum_ps = np.zeros(NCH); n_ps = 0
sum_z2 = np.zeros(NCH); sum_absz = np.zeros(NCH); n_within = np.zeros(NCH); n_z = 0

for j in JOBS:
    xs = np.load(f"{D}/x_samples_job_{j}.npy", mmap_mode="r")   # (n,500,128,128,5)
    tx = np.load(f"{D}/true_x_job_{j}.npy")                     # (n,128,128,5)
    n = xs.shape[0]
    for i in range(n):
        s = np.asarray(xs[i], np.float64)        # (500,128,128,5)
        pm = s.mean(0); ps = s.std(0)            # (128,128,5)
        z = (tx[i] - pm) / ps
        sum_ps += ps.reshape(-1, NCH).mean(0); n_ps += 1
        zc = z.reshape(-1, NCH)
        sum_z2 += (zc**2).mean(0)
        sum_absz += np.abs(zc).mean(0)
        n_within += (np.abs(zc) < 1).mean(0)
        n_z += 1

s_model = sum_ps / n_ps
rms_z = np.sqrt(sum_z2 / n_z)
mean_absz = sum_absz / n_z
frac_within = n_within / n_z
s_post_ref = slsst * fstd / np.sqrt(slsst**2 + fstd**2)

print(f"\n{'ch':<3}{'s_model':>10}{'sigma_lsst':>12}{'s_post_ref':>12}"
      f"{'s_mod/lsst':>12}{'s_mod/ref':>11}{'RMS(z)':>9}{'E|z|':>8}{'P(|z|<1)':>10}")
for c in range(NCH):
    print(f"{c:<3}{s_model[c]:>10.4f}{slsst[c]:>12.4f}{s_post_ref[c]:>12.4f}"
          f"{s_model[c]/slsst[c]:>12.3f}{s_model[c]/s_post_ref[c]:>11.3f}"
          f"{rms_z[c]:>9.2f}{mean_absz[c]:>8.2f}{frac_within[c]:>10.2f}")

print(f"\nPooled: RMS(z)={np.sqrt(sum_z2.sum()/(n_z*NCH)):.2f}  "
      f"(calibrated ~1.0; >>1 = overconfident)  "
      f"mean s_model/s_post_ref={np.mean(s_model/s_post_ref):.3f}  "
      f"(<<1 = posterior narrower than Bayesian reference)")
