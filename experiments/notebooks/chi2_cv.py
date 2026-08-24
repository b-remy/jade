"""Chi-square diagnostics for the cosmic-variance comparison.

Loads the cache written by ``plot_ps_cosmic_variance.py`` (drawn fields, JADE
posterior maps, and all per-realization power spectra) and computes the
chi-square diagnostics WITHOUT re-running the GPU sampling. Iterate on the
metrics here freely -- it is pure numpy/scipy and runs in seconds on CPU.

Usage:
    python chi2_cv.py
    CACHE_TAG=N100x10_T100_JADE_B_16_ema_best python chi2_cv.py
    N_BAND=10 python chi2_cv.py        # coarser ell-bands for the covariance
"""

import os
import glob

import numpy as np

save_dir = "amortized_cosmic_variance"

# ---------------------------------------------------------------------------
# Load the cache. If CACHE_TAG is unset, pick the most recently written cache.
# ---------------------------------------------------------------------------
CACHE_TAG = os.environ.get("CACHE_TAG")
if CACHE_TAG:
    CACHE_PATH = os.path.join(save_dir, f"cv_cache_{CACHE_TAG}.npz")
else:
    candidates = sorted(
        glob.glob(os.path.join(save_dir, "cv_cache_*.npz")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise SystemExit(
            f"No cache found in {save_dir}/cv_cache_*.npz -- run "
            f"plot_ps_cosmic_variance.py first to generate one."
        )
    CACHE_PATH = candidates[-1]

print(f"Loading cache: {CACHE_PATH}")
c = np.load(CACHE_PATH, allow_pickle=True)
clean_fields = c["clean_fields"]
noisy_obs = c["noisy_obs"]
all_post_maps = c["all_post_maps"]
ell = c["ell"]
truth_auto = c["truth_auto"]
truth_cross = c["truth_cross"]
post_auto = c["post_auto"]
post_cross = c["post_cross"]
sigma_lsst = c["sigma_lsst"]
N_REAL = int(c["N_REAL"])
N_SAMPLES_PER_OBS = int(c["N_SAMPLES_PER_OBS"])
N_TRUTH = int(c["N_TRUTH"])
print(f"  N_REAL={N_REAL}  N_SAMPLES_PER_OBS={N_SAMPLES_PER_OBS}  N_TRUTH={N_TRUTH}  "
      f"params={c['PARAMS_NAME']}")
print(f"  post_auto {post_auto.shape}  truth_auto {truth_auto.shape}  "
      f"all_post_maps {all_post_maps.shape}")

ps_auto_truth = truth_auto.mean(0)
ps_auto_mean = post_auto.mean(0)

# ---------------------------------------------------------------------------
# Quick scalar diagnostic: mean |relative bias| of the auto spectra.
# ---------------------------------------------------------------------------
rel = (ps_auto_truth - ps_auto_mean) / ps_auto_truth
print("\nAverage |relative bias| per auto bin:")
for i in range(5):
    print(f"  bin {i}: {np.mean(np.abs(rel[i])):.4f}")
print(f"Overall: {np.mean(np.abs(rel)):.4f}")

# ---------------------------------------------------------------------------
# Full-covariance chi-square of the auto spectra (JADE mean vs truth mean).
#
# Test: is the JADE posterior-mean C_ell statistically consistent with the
# truth-mean C_ell, accounting for the *correlated* scatter across ell-bins?
#   chi^2 = r^T Cov^{-1} r,   r = <C_ell>_JADE - <C_ell>_truth
# Coarse-rebin into N_BAND log-spaced bands so p = 5*N_BAND << N_jade and the
# Hartlap factor stays close to 1.  Cov(r) = C_single * (1/N_jade + 1/N_truth).
# ---------------------------------------------------------------------------
N_BAND = int(os.environ.get("N_BAND", 16))
N_jade = post_auto.shape[0]
N_truth = truth_auto.shape[0]
n_ell = post_auto.shape[2]

band_edges = np.unique(np.geomspace(1, n_ell, N_BAND + 1).round().astype(int))
band_edges[0], band_edges[-1] = 0, n_ell
n_band = len(band_edges) - 1


def rebin_ell(arr):
    """Average fine ell-bins into coarse bands. arr: (..., n_ell) -> (..., n_band)."""
    return np.stack(
        [arr[..., band_edges[b]:band_edges[b + 1]].mean(axis=-1) for b in range(n_band)],
        axis=-1,
    )


jade_vec = rebin_ell(post_auto).reshape(N_jade, -1)
truth_vec = rebin_ell(truth_auto).reshape(N_truth, -1)
p = jade_vec.shape[1]

r = jade_vec.mean(0) - truth_vec.mean(0)
C_single = np.cov(jade_vec, rowvar=False)
cov_diff = C_single * (1.0 / N_jade + 1.0 / N_truth)

try:
    from scipy.stats import chi2 as _chi2dist
except Exception:
    _chi2dist = None

if N_jade <= p + 2:
    print(f"[chi2] WARNING: N_jade={N_jade} <= p+2={p + 2}; covariance not invertible. "
          f"Lower N_BAND.")
else:
    hartlap = (N_jade - p - 2) / (N_jade - 1)
    cinv = np.linalg.inv(cov_diff) * hartlap
    chi2 = float(r @ cinv @ r)
    dof = p
    pval = float(_chi2dist.sf(chi2, dof)) if _chi2dist is not None else float("nan")
    cond = float(np.linalg.cond(cov_diff))
    print(f"\nFull-covariance chi^2 (auto spectra, {n_band} bands x 5 z-bins):")
    print(f"  Hartlap factor = {hartlap:.3f}   cov cond. number = {cond:.2e}")
    print(f"  chi^2 = {chi2:.1f}   dof = {dof}   chi^2/dof = {chi2 / dof:.3f}   "
          f"p-value = {pval:.3g}")

    jade_z = rebin_ell(post_auto)
    truth_z = rebin_ell(truth_auto)
    print("  Per z-bin chi^2/dof (dof = n_band):")
    for i in range(5):
        ri = jade_z[:, i].mean(0) - truth_z[:, i].mean(0)
        Ci = np.cov(jade_z[:, i], rowvar=False) * (1.0 / N_jade + 1.0 / N_truth)
        hi = (N_jade - n_band - 2) / (N_jade - 1)
        chi2_i = float(ri @ (np.linalg.inv(Ci) * hi) @ ri)
        print(f"    bin {i}: chi^2={chi2_i:.1f}  chi^2/dof={chi2_i / n_band:.3f}")

# ---------------------------------------------------------------------------
# Test C: per-observation calibration chi-square (power spectrum).
#
# For EACH observation, is the underlying truth field's power spectrum a
# plausible draw from JADE's posterior predictive? Does NOT divide by 1/N, so a
# well-calibrated posterior gives chi^2/dof ~ 1.  Reports per-obs sigma and a
# pooled-sigma variant (more stable given only N_S draws/obs).
# ---------------------------------------------------------------------------
N_S = N_SAMPLES_PER_OBS
post_zb = rebin_ell(post_auto).reshape(N_REAL, N_S, 5, n_band)  # (obs, draw, z, band)
truth_obs = rebin_ell(truth_auto[:N_REAL])                     # (obs, z, band)

post_mean_o = post_zb.mean(1)
post_var_o = post_zb.var(1, ddof=1)
resid_o = truth_obs - post_mean_o
pooled_var = post_var_o.mean(0)

eps = 1e-30
chi2_per_obs = (resid_o**2 / (post_var_o + eps)).sum(axis=(1, 2))
chi2_per_obs_pool = (resid_o**2 / (pooled_var[None] + eps)).sum(axis=(1, 2))
dof_joint = 5 * n_band

print(f"\nTest C: per-observation calibration chi^2 "
      f"({N_REAL} obs, {N_S} draws/obs, {n_band} bands x 5 z-bins, dof={dof_joint}):")
print(f"  per-obs sigma   : mean chi^2/dof = {chi2_per_obs.mean() / dof_joint:.3f}   "
      f"median = {np.median(chi2_per_obs) / dof_joint:.3f}")
print(f"  pooled sigma    : mean chi^2/dof = {chi2_per_obs_pool.mean() / dof_joint:.3f}   "
      f"median = {np.median(chi2_per_obs_pool) / dof_joint:.3f}")
print("  Per z-bin mean chi^2/dof (dof = n_band, pooled sigma):")
for i in range(5):
    cc = (resid_o[:, i]**2 / (pooled_var[i][None] + eps)).sum(axis=1)
    print(f"    bin {i}: mean = {cc.mean() / n_band:.3f}   median = {np.median(cc) / n_band:.3f}")

# ---------------------------------------------------------------------------
# Pixel-level data-residual chi-square (Gaussian noise check).
#
# obs = kappa_true + noise,  noise ~ N(0, sigma_lsst_z^2) i.i.d. per pixel.
# A good posterior sample x_s makes (obs - x_s) just that Gaussian noise:
#   chi^2(x_s) = sum_{pix,z} ( (obs - x_s) / sigma_lsst_z )^2,  dof = Npix * 5
# Perfect (x_s == kappa_true): chi^2/dof = 1 +/- sqrt(2/dof).
#   <1 -> samples overfit the noise (too rough);  >1 -> samples miss the data
#   (posterior too tight / wrong, or noise underestimated).
# Gaussianity checked via standardized residual z=resid/sigma: <z>~0, std(z)~1.
# ---------------------------------------------------------------------------
obs_rep = np.repeat(noisy_obs, N_SAMPLES_PER_OBS, axis=0)
sig_z = np.asarray(sigma_lsst).reshape(1, 1, 1, -1)
zres = (all_post_maps - obs_rep) / sig_z
n_pix = all_post_maps.shape[1] * all_post_maps.shape[2]

chi2_pix = (zres**2).sum(axis=(1, 2, 3))
dof_pix = n_pix * 5
chi2_pix_z = (zres**2).sum(axis=(1, 2))

print(f"\nPixel data-residual chi^2 (obs - x_s, {all_post_maps.shape[0]} samples, "
      f"dof = Npix*5 = {dof_pix}):")
print(f"  overall: mean chi^2/dof = {chi2_pix.mean() / dof_pix:.4f}   "
      f"median = {np.median(chi2_pix) / dof_pix:.4f}   "
      f"sample-to-sample std = {chi2_pix.std() / dof_pix:.4f}")
print(f"  ideal  : 1.0000 +/- {np.sqrt(2.0 / dof_pix):.4f}")
print("  Per z-bin chi^2/dof   |   <z> (->0)   std(z) (->1):")
for i in range(5):
    cz = chi2_pix_z[:, i].mean() / n_pix
    zmean = float(zres[..., i].mean())
    zstd = float(zres[..., i].std())
    print(f"    bin {i}: chi^2/dof = {cz:.4f}   |   <z> = {zmean:+.4f}   std(z) = {zstd:.4f}")

# ---------------------------------------------------------------------------
# One-point PDF relative error per z-bin (same KDE convention as the figure in
# plot_amortized.py: per-realization gaussian_kde on a grid set by the truth's
# [0.1, 99.9] percentiles; truth PDF = mean over fields, JADE PDF = mean over
# sample maps).  Relative error mirrors the power-spectrum one:
#     rel(kappa) = | p_truth - p_JADE | / p_truth
# averaged over the grid, restricted to the well-supported region
# (p_truth > 1% of its peak) so the near-zero tails don't blow up the ratio.
# Also reports the division-free L1 distance  int |p_truth - p_JADE| dkappa
# (= 2 x total-variation distance; 0 = identical, <= 2).
# ---------------------------------------------------------------------------
from scipy.stats import gaussian_kde

N_KDE = min(64, all_post_maps.shape[0])   # JADE maps used for the KDE band
GRID = 400
SUPP = 1e-2                               # support threshold (fraction of peak)

print("\nOne-point PDF relative error per z-bin "
      f"(KDE, {clean_fields.shape[0]} truth fields, {N_KDE} JADE maps):")
rel_all = []     # pooled |rel| over all supported (z, grid) points, like the PS overall
l1_all = []
for i in range(5):
    truth_px = clean_fields[..., i].reshape(clean_fields.shape[0], -1)  # (n_truth, npix)
    lo, hi = np.percentile(truth_px.ravel(), [0.1, 99.9])
    pad = 0.1 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, GRID)

    pdf_truth = np.mean([gaussian_kde(truth_px[t])(grid) for t in range(truth_px.shape[0])], axis=0)
    pdf_jade = np.mean([gaussian_kde(all_post_maps[s, :, :, i].ravel())(grid)
                        for s in range(N_KDE)], axis=0)

    m = pdf_truth > SUPP * pdf_truth.max()           # well-supported region
    rel = np.abs(pdf_truth[m] - pdf_jade[m]) / pdf_truth[m]
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    l1 = _trap(np.abs(pdf_truth - pdf_jade), grid)  # division-free
    rel_all.append(rel)
    l1_all.append(l1)
    print(f"    bin {i}: mean|rel| = {rel.mean():.4f}   median|rel| = {np.median(rel):.4f}"
          f"   L1 = {l1:.4f}")
rel_all = np.concatenate(rel_all)
print(f"Overall: mean|rel| = {rel_all.mean():.4f}   median|rel| = {np.median(rel_all):.4f}"
      f"   mean L1 = {np.mean(l1_all):.4f}")
