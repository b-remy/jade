"""Paper figures from the arrays written by sample_posterior.py.

  Fig 2  joint_posterior_samples  observation, truth, and two posterior draws
  Fig 3  power-spectra-posterior  auto- and cross-C_l over tomographic bins
  Fig 5  contour_plot             1D/2D marginals against the NUTS chain
  Fig 6  one-point-function       kappa PDF per bin (appendix)

``--diagnostics`` adds the relative-C_l and cross-correlation panels, which are
not in the paper.
"""

import argparse
import itertools
import os

import astropy.units as u
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from scipy.stats import gaussian_kde
from tqdm import tqdm

from lenstools import ConvergenceMap
from getdist import MCSamples, plots

from jade.paths import FIGURES_DIR

# Computer Modern for anything inside $...$, bundled with matplotlib, so the
# labels match pdflatex without a system TeX install.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

NAMES = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]
TEX_NAMES = [r"\Omega_c", r"\Omega_b", r"\sigma_8", r"h_0", r"n_s", r"w_0"]
N_BINS = 5
MAP_SIZE = 5
L_EDGES = np.linspace(500, 4608.0, 128)

FONTSIZE_TEXT = 32
FONTSIZE_TICKS = 20
FONTSIZE_LEGEND = 18
TICK_MAJOR = 8
TICK_MINOR = 4
TICK_WIDTH = 1.4


def save(fig, out_dir, name, **kwargs):
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, **kwargs)
    print(f"wrote {name}.{{png,pdf}}")
    plt.close(fig)


def fill_lower_diag(array, nl):
    n = int(np.sqrt(len(array) * 2)) + 1
    mask = np.arange(n)[:, None] > np.arange(n)
    out = np.zeros((n, n, nl))
    out[np.stack(mask, axis=1)] = array
    return out.T


def compute_ps(map_a, map_b):
    """Auto- and cross-spectra of two (128, 128, 5) convergence stacks."""
    p_cross = []
    for i, j in itertools.combinations(range(N_BINS), 2):
        ell, ps = ConvergenceMap(map_a[:, :, i], angle=MAP_SIZE * u.deg).cross(
            ConvergenceMap(map_b[:, :, j], angle=MAP_SIZE * u.deg),
            l_edges=L_EDGES)
        p_cross.append(ps)
    ps_cross = fill_lower_diag(np.array(p_cross), len(L_EDGES) - 1)

    ps_auto = []
    for i in range(N_BINS):
        ell, pi = ConvergenceMap(map_a[:, :, i], angle=MAP_SIZE * u.deg).cross(
            ConvergenceMap(map_b[:, :, i], angle=MAP_SIZE * u.deg),
            l_edges=L_EDGES)
        ps_auto.append(pi)
    return ell, np.array(ps_auto), ps_cross


def style_triangle_axes(ax, log_x=True):
    if log_x:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", length=TICK_MAJOR, width=TICK_WIDTH)
    ax.tick_params(which="minor", length=TICK_MINOR, width=TICK_WIDTH)


def fig_contours(d, out_dir):
    """Figure 5."""
    mcmc = MCSamples(samples=d["mcmc_samples"], names=NAMES, label="MCMC")
    jade = MCSamples(samples=d["theta_samples"], names=NAMES,
                     label="JADE (our work)")

    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 34
    g.settings.axes_labelsize = 38
    g.settings.legend_fontsize = 24
    g.settings.lab_fontsize = 38
    g.settings.tight_layout = True

    g.triangle_plot(
        [jade, mcmc],
        NAMES,
        markers=d["theta_truth"],
        marker_args={"lw": 1},
        filled=[True, False],
        line_args=[
            {"ls": "-", "color": "#d06e99ff"},
            {"ls": "--", "color": "black"},
        ],
        alpha=[1.0, 0.1],
        contour_colors=["#d06e99ff", "black"],
        contour_ls=["-", "--"],
        contour_lws=[4.0, 4.0],
    )

    # getdist's axes_fontsize sometimes only reaches the major numeric labels.
    for row in g.subplots:
        for ax in row:
            if ax is not None:
                ax.tick_params(axis="both", which="major", labelsize=28,
                               length=8, width=1.2)
                ax.tick_params(axis="both", which="minor", length=4, width=1.0)

    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(out_dir, f"contour_plot.{ext}"))
    print("wrote contour_plot.{png,pdf}")
    plt.close("all")


def fig_joint_samples(d, out_dir, n_instances=2):
    """Figure 2."""
    obs = d["observation"]
    ref = d["reference_field"]
    theta_truth = d["theta_truth"]
    kappa = d["kappa_samples"]
    theta = d["theta_samples"]
    cmap = "magma"

    vmin = [ref[..., c].min() for c in range(N_BINS)]
    vmax = [ref[..., c].max() for c in range(N_BINS)]

    fig = plt.figure(figsize=(14, 2.5 * (n_instances + 2)))
    gs = fig.add_gridspec(
        n_instances + 2, N_BINS + 2,
        width_ratios=[0.1, 0.6] + [1] * N_BINS, hspace=0.0, wspace=0.0,
    )

    def row_label(row, text):
        ax = fig.add_subplot(gs[row, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, text, fontsize=16, ha="center", va="center",
                rotation=90)

    def theta_column(row, values):
        ax = fig.add_subplot(gs[row, 1])
        ax.axis("off")
        ax.text(0.5, 0.5,
                "\n".join(f"${TEX_NAMES[i]}$: {float(values[i]):.3f}"
                          for i in range(6)),
                fontsize=14, ha="center", va="center")

    # Row 0: the noisy observation, on its own colour scale per bin.
    obs_axes = []
    row_label(0, "Observation")
    fig.add_subplot(gs[0, 1]).axis("off")
    for c in range(N_BINS):
        ax = fig.add_subplot(gs[0, c + 2])
        im = ax.imshow(obs[..., c], cmap=cmap)
        ax.axis("off")
        obs_axes.append((ax, im, c))

    # Row 1: noiseless reference field at Planck15.
    row_label(1, "Ground Truth")
    theta_column(1, theta_truth)
    for c in range(N_BINS):
        ax = fig.add_subplot(gs[1, c + 2])
        ax.imshow(ref[..., c], cmap=cmap, vmin=vmin[c], vmax=vmax[c])
        ax.axis("off")

    # Rows 2+: posterior kappa draws with their inferred cosmology.
    last_axes = []
    for n in range(n_instances):
        row_label(n + 2, f"Sample {n + 1}")
        theta_column(n + 2, theta[n])
        for c in range(N_BINS):
            ax = fig.add_subplot(gs[n + 2, c + 2])
            im = ax.imshow(kappa[n, ..., c], cmap=cmap,
                           vmin=vmin[c], vmax=vmax[c])
            ax.axis("off")
            if n == n_instances - 1:
                last_axes.append((ax, im))

    for ax, im, c in obs_axes:
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y1 + 0.002, pos.width, 0.015])
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=12)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.set_title(f"Bin {c}", fontsize=14, pad=10)

    for ax, im in last_axes:
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - 0.017, pos.width, 0.015])
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=12)

    save(fig, out_dir, "joint_posterior_samples",
         bbox_inches="tight", pad_inches=0)


def spectra(d, n_samples):
    """Truth spectra and the per-draw posterior spectra."""
    kappa = d["kappa_samples"]
    ell, ps_auto, ps_cross = compute_ps(d["reference_field"],
                                        d["reference_field"])

    n = min(n_samples, len(kappa))
    auto, cross = [], []
    for s in tqdm(range(n), desc="computing ps"):
        _, a, c = compute_ps(kappa[s], kappa[s])
        auto.append(a)
        cross.append(c)
    return ell, ps_auto, ps_cross, np.array(auto), np.array(cross)


def _triangle_grid(plot_diag, plot_off, ylabel, out_dir, name, log_y):
    fig, ax = plt.subplots(N_BINS, N_BINS, figsize=(10, 10))
    for i in range(N_BINS):
        for j in range(N_BINS):
            if j > i:
                ax[i, j].axis("off")
            else:
                (plot_diag if i == j else plot_off)(ax[i, j], i, j)
                ax[i, j].set_xscale("log")
                if log_y:
                    ax[i, j].set_yscale("log")
                    ax[i, j].yaxis.set_major_locator(
                        LogLocator(base=10.0, numticks=4))
                    ax[i, j].yaxis.set_minor_formatter(NullFormatter())
                else:
                    ax[i, j].yaxis.set_major_locator(MaxNLocator(nbins=4))
                style_triangle_axes(ax[i, j])

            if i == N_BINS - 1:
                ax[i, j].tick_params(axis="x", labelsize=FONTSIZE_TICKS)
            else:
                ax[i, j].tick_params(axis="x", labelbottom=False)
            if j == 0:
                ax[i, j].tick_params(axis="y", labelsize=FONTSIZE_TICKS)
            else:
                ax[i, j].tick_params(axis="y", labelleft=False)

    fig.supxlabel(r"$\ell$", fontsize=FONTSIZE_TEXT)
    fig.supylabel(ylabel, fontsize=FONTSIZE_TEXT, x=-0.02)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.88, 0.88),
               fontsize=FONTSIZE_LEGEND)
    save(fig, out_dir, name, bbox_inches="tight", pad_inches=0.05)


def fig_power_spectra(ell, ps_auto, ps_cross, auto, cross, out_dir):
    """Figure 3."""
    a_mean, a_std = auto.mean(0), auto.std(0)
    c_mean, c_std = cross.mean(0), cross.std(0)

    def diag(ax, i, _j):
        ax.plot(ell, ps_auto[i], label="Ground truth", color="k", alpha=1.0)
        ax.plot(ell, a_mean[i], color="tab:blue", label="JADE samples")
        ax.fill_between(ell, a_mean[i] - a_std[i], a_mean[i] + a_std[i],
                        color="tab:blue", alpha=0.3)
        ax.set_xlim(ell.min(), ell.max())

    def off(ax, i, j):
        ax.plot(ell, ps_cross[:, i, j], color="k")
        ax.plot(ell, c_mean[:, i, j], color="tab:blue", alpha=1.0)
        ax.fill_between(ell, c_mean[:, i, j] - c_std[:, i, j],
                        c_mean[:, i, j] + c_std[:, i, j],
                        color="tab:blue", alpha=0.3)
        ax.set_xlim(ell.min(), ell.max())

    _triangle_grid(diag, off, r"$\mathcal{C}_\ell$", out_dir,
                   "power-spectra-posterior", log_y=True)

    err = (ps_auto - a_mean) / ps_auto
    print("Average relative error per auto bin:")
    for i, e in enumerate(np.mean(np.abs(err), axis=1)):
        print(f"  bin {i}: {e:.4f}")
    print(f"Overall average relative error (auto): {np.mean(np.abs(err)):.4f}")


def fig_one_point(d, out_dir, n_samples=64):
    """Figure 6 (appendix)."""
    truth = d["reference_field"]
    kappa = d["kappa_samples"]
    n = min(n_samples, len(kappa))

    fig, ax = plt.subplots(1, N_BINS, figsize=(18, 4), sharey=False)
    for i in range(N_BINS):
        vals = np.asarray(truth[..., i]).ravel()
        lo, hi = np.percentile(vals, [0.1, 99.9])
        pad = 0.1 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 400)

        pdf_truth = gaussian_kde(vals)(grid)
        pdf_samples = np.stack([
            gaussian_kde(np.asarray(kappa[s, :, :, i]).ravel())(grid)
            for s in range(n)
        ])
        mean, std = pdf_samples.mean(0), pdf_samples.std(0)

        ax[i].plot(grid, pdf_truth, color="k", label="Truth")
        ax[i].plot(grid, mean, color="tab:blue", label="JADE samples")
        ax[i].fill_between(grid, mean - std, mean + std,
                           color="tab:blue", alpha=0.3)
        ax[i].set_title(f"bin {i}", fontsize=FONTSIZE_TICKS)
        ax[i].tick_params(axis="both", labelsize=FONTSIZE_TICKS)
        ax[i].xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax[i].set_xlim(grid[0], grid[-1])

    fig.supxlabel(r"$\kappa$", fontsize=FONTSIZE_TICKS, y=-0.08)
    fig.supylabel(r"$p(\kappa)$", fontsize=FONTSIZE_TICKS)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(labels), fontsize=18, frameon=False)
    save(fig, out_dir, "one-point-function", bbox_inches="tight", pad_inches=0)


def fig_relative_ps(ell, ps_auto, ps_cross, auto, cross, out_dir):
    """Not in the paper: C_l residuals against the noiseless truth."""
    rel_auto = (ps_auto[None] - auto) / ps_auto[None]
    safe = np.where(ps_cross[None] == 0, 1.0, ps_cross[None])
    rel_cross = (ps_cross[None] - cross) / safe

    a_mean, a_std = rel_auto.mean(0), rel_auto.std(0)
    c_mean, c_std = rel_cross.mean(0), rel_cross.std(0)

    def diag(ax, i, _j):
        ax.axhline(0, color="k", alpha=1.0, label="Ground truth")
        ax.plot(ell, a_mean[i], color="tab:blue", label="JADE sample")
        ax.fill_between(ell, a_mean[i] - a_std[i], a_mean[i] + a_std[i],
                        color="tab:blue", alpha=0.3)
        ax.set_xlim(ell.min(), ell.max())

    def off(ax, i, j):
        ax.axhline(0, color="k")
        ax.plot(ell, c_mean[:, i, j], color="tab:blue", alpha=1.0)
        ax.fill_between(ell, c_mean[:, i, j] - c_std[:, i, j],
                        c_mean[:, i, j] + c_std[:, i, j],
                        color="tab:blue", alpha=0.3)
        ax.set_xlim(ell.min(), ell.max())

    _triangle_grid(
        diag, off,
        r"$(\mathcal{C}_\ell^{\rm truth}-\mathcal{C}_\ell^{\rm sample})"
        r"/\mathcal{C}_\ell^{\rm truth}$",
        out_dir, "power-spectra-posterior-relative", log_y=False)


def fig_cross_correlation(d, ell, ps_auto, out_dir):
    """Not in the paper: r(l) between the posterior mean field and the truth."""
    truth = d["reference_field"]
    mean_field = np.asarray(d["kappa_samples"]).mean(axis=0)

    r = np.zeros((N_BINS, len(ell)))
    for i in range(N_BINS):
        t_map = ConvergenceMap(truth[:, :, i], angle=MAP_SIZE * u.deg)
        m_map = ConvergenceMap(mean_field[:, :, i], angle=MAP_SIZE * u.deg)
        _, p_mt = m_map.cross(t_map, l_edges=L_EDGES)
        _, p_mm = m_map.cross(m_map, l_edges=L_EDGES)
        r[i] = p_mt / np.sqrt(p_mm * ps_auto[i])

    fig, ax = plt.subplots(1, N_BINS, figsize=(18, 4), sharey=True)
    for i in range(N_BINS):
        ax[i].axhline(1.0, color="k", label="Perfect correlation")
        ax[i].plot(ell, r[i], color="tab:blue", label="Posterior mean field")
        ax[i].set_xscale("log")
        ax[i].set_xlim(ell.min(), ell.max())
        ax[i].set_title(f"bin {i + 1}", fontsize=FONTSIZE_TEXT)
        ax[i].tick_params(axis="both", labelsize=FONTSIZE_TICKS)
        ax[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax[i].xaxis.set_minor_formatter(NullFormatter())

    fig.supxlabel(r"$\ell$", fontsize=FONTSIZE_TEXT)
    fig.supylabel(r"$r(\ell)$", fontsize=FONTSIZE_TEXT)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(labels), fontsize=18, frameon=False)
    save(fig, out_dir, "cross-correlation-coefficient",
         bbox_inches="tight", pad_inches=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(FIGURES_DIR / "posterior.npz"),
                        help="Output of sample_posterior.py.")
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    parser.add_argument("--ps-samples", type=int, default=64,
                        help="Posterior draws used for the spectra and PDFs.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Also make the two panels not in the paper.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = dict(np.load(args.samples))
    print(f"{args.samples}: {len(d['kappa_samples'])} posterior draws")

    fig_contours(d, args.out_dir)
    fig_joint_samples(d, args.out_dir)

    ell, ps_auto, ps_cross, auto, cross = spectra(d, args.ps_samples)
    fig_power_spectra(ell, ps_auto, ps_cross, auto, cross, args.out_dir)
    fig_one_point(d, args.out_dir, n_samples=args.ps_samples)

    if args.diagnostics:
        fig_relative_ps(ell, ps_auto, ps_cross, auto, cross, args.out_dir)
        fig_cross_correlation(d, ell, ps_auto, args.out_dir)


if __name__ == "__main__":
    main()
