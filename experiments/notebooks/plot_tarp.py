from tarp import get_tarp_coverage
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Paper-style typography without a system TeX install. ``mathtext.fontset='cm'``
# uses Computer Modern (bundled with matplotlib) for everything inside $...$,
# while plain text falls back to whatever serif font is available (DejaVu Serif
# on most clusters). Matches plot_amortized.py and plot_tarp_conditional.py.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

samples = []
cosmo = []
for i in range(1):
    samples.append(np.load(f'tarp_results/256/cosmo_samples_job_{i}.npy'))
    cosmo.append(np.load(f'tarp_results/256/true_cosmo_job_{i}.npy'))
    
samples = np.concatenate(samples,0)
cosmo = np.concatenate(cosmo, 0)

samples_tarp = np.transpose(samples, [1,0,2])
# samples: (n_samples, n_sims, n_dims)
cosmo_tarp = cosmo

print("samples_tarp shape:", samples_tarp.shape)
print("cosmo_tarp shape:", cosmo_tarp.shape)

ecp_bootstrap, alpha_bootstrap = get_tarp_coverage(samples_tarp, cosmo_tarp, 
                                                   references = "random", metric = "euclidean", 
                                                   norm = True, bootstrap=True,
                                                  num_bootstrap=100)

k_sigma = [1, 2, 3]

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
ax.plot(alpha_bootstrap, ecp_bootstrap.mean(axis=0), label='TARP', color='tab:blue')
for k in k_sigma:
    ax.fill_between(alpha_bootstrap, 
                    ecp_bootstrap.mean(axis=0) - k * ecp_bootstrap.std(axis=0), ecp_bootstrap.mean(axis=0) + k * ecp_bootstrap.std(axis=0), 
                    alpha = 0.2, color='tab:blue')
ax.legend()
ax.set_ylabel("Expected Coverage")
ax.set_xlabel("Credibility Level")

plt.savefig('tarp_results/tarp.png')