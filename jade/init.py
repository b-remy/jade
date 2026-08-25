import jax_cosmo as jc
import numpy as np
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.redshift import subdivide

# lognormal theta mean and std
THETA_MEAN = np.array([0.30303165, 0.04918509, 0.83073664, 0.67226803, 0.964793, -1.1082776])
THETA_STD = np.array([0.17090547, 0.00601253, 0.1402754, 0.06292731, 0.08028358, 0.46022922])

# lognormal field mean and std
FIELD_MEAN = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
FIELD_STD = np.array([0.00468968, 0.01021258, 0.01501551, 0.02049607, 0.02891253])

# GRF g mean and std
GRF_MEAN = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
GRF_STD = np.array([0.48319262, 0.389879, 0.32836843, 0.27031878, 0.21253702])

# lsst noise level

map_size = 5
N = 128
pix_area = (map_size * 60 / N) ** 2
sigma_e = 0.26
nz = jc.redshift.smail_nz(
    config_lsst_y_10.a, config_lsst_y_10.b, config_lsst_y_10.z0, gals_per_arcmin2=config_lsst_y_10.gals_per_arcmin2
)
nz_bins = subdivide(nz, nbins=config_lsst_y_10.nbins, zphot_sigma=0.05)
tracer = jc.probes.WeakLensing(nz_bins, sigma_e=config_lsst_y_10.sigma_e)

sigma_lsst = np.sqrt(sigma_e**2 / (np.array([b.gals_per_arcmin2 for b in nz_bins]) * pix_area))
