"""
Generate a fresh (truth κ, noisy obs) pair from the LSST log-normal simulator,
run JADE on the noisy obs, and compare the C_ℓ of the posterior κ samples to
the C_ℓ of the very same noiseless truth κ that produced the obs.

Unlike plot_amortized.py — which compares posterior C_ℓ against a separately
re-traced Planck15 reference — this script draws (truth, obs) in one place so
the comparison is mechanically unambiguous: obs = truth_κ + noise (same key,
toggling ``with_noise``; numpyro's per-site sub-keys give bit-identical κ in
both traces).
"""

import argparse
import itertools
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

from flax import nnx
import jax_cosmo as jc
from numpyro.handlers import condition, seed, trace
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal

from jade.nn_hybrid import JADE_B_16
from jade.init import FIELD_MEAN, FIELD_STD, THETA_MEAN, THETA_STD
from jade.flow import Denoiser
from jade.utils import load_model
from jade.sampling import HeunSampler

import astropy.units as u
from lenstools import ConvergenceMap
from tqdm import tqdm


class AdjointMatchingSDESampler(nnx.Module):
    r"""Memoryless-schedule generative SDE sampler for the linear-interpolant flow.

    dX = b dt + sigma dB, Euler-Maruyama forward in t from 0 (noise) to 1 (data):
        b(x,t)   = (1+g) v(x,t) - g * x / t
        sigma(t) = sqrt(2 g (1-t)/t)
    g=1 is the memoryless schedule; g=0 recovers the deterministic Euler ODE.
    Drop-in for HeunSampler (same call signature). See plot_amortized_sde.py
    (Domingo-Enrich et al. 2024, Eqs. 10-11).
    """

    def __init__(self, model, num_steps: int = 1000, t_eps: float = 0.05,
                 noise_scale: float = 1.0):
        self.model = model
        self.num_steps = num_steps
        self.t_eps = t_eps
        self.g = noise_scale

    def __call__(self, x0, cosmo0, cond=None, key=None):
        ts = jnp.linspace(0.0, 1.0, self.num_steps + 1)
        keys = jax.random.split(key, self.num_steps)
        g = self.g

        def em_step(carry, inp):
            xt, cosmot = carry
            t, t_next, k = inp
            dt = t_next - t

            v_x, v_cosmo = self.model.v_pred(xt, cosmot, t, cond=cond, train=False)

            t_lo = jnp.clip(t, a_min=self.t_eps)
            kappa = 1.0 / t_lo
            eta = (1.0 - t) / t_lo
            sigma = jnp.sqrt(jnp.clip(2.0 * g * eta, a_min=0.0))

            b_x = (1.0 + g) * v_x - g * kappa * xt
            b_cosmo = (1.0 + g) * v_cosmo - g * kappa * cosmot

            k_x, k_c = jax.random.split(k, 2)
            noise_x = jax.random.normal(k_x, shape=xt.shape)
            noise_cosmo = jax.random.normal(k_c, shape=cosmot.shape)

            sqrt_dt = jnp.sqrt(jnp.clip(dt, a_min=0.0))
            xt_n = xt + b_x * dt + sigma * sqrt_dt * noise_x
            cosmot_n = cosmot + b_cosmo * dt + sigma * sqrt_dt * noise_cosmo
            return (xt_n, cosmot_n), None

        (xt, cosmot), _ = jax.lax.scan(
            em_step, (x0, cosmo0), (ts[:-1], ts[1:], keys)
        )
        return xt, cosmot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/u/bremy/repos/jade/experiments/wandb/run-20260507_170014-7hnur00g/files/checkpoints",
    )
    parser.add_argument("--ckpt-tag", default="JADE_B_16_latest")
    parser.add_argument("--seed", type=int, default=42,
                        help="Key seed for the simulator (truth κ + obs).")
    parser.add_argument("--sampler-seed", type=int, default=43,
                        help="Disjoint key seed for the JADE sampler.")
    parser.add_argument("--n-samples", type=int, default=512,
                        help="Total posterior κ samples (rounded up to batch_size).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Sampler steps; default 200 (ODE) / 1000 (SDE).")
    parser.add_argument("--sampler", choices=["ode", "sde"], default="ode",
                        help="ode=deterministic Heun; sde=memoryless Adjoint-Matching SDE.")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="SDE stochasticity knob g (g=1 memoryless; only for --sampler sde).")
    parser.add_argument("--save-dir", default="amortized")
    args = parser.parse_args()

    num_steps = args.num_steps if args.num_steps is not None else (
        1000 if args.sampler == "sde" else 200)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"sampler={args.sampler}  num_steps={num_steps}"
          + (f"  g={args.noise_scale}" if args.sampler == "sde" else ""))

    # -- Load JADE
    cfg, states = load_model(args.checkpoint, args.ckpt_tag)
    model = JADE_B_16(
        rngs=nnx.Rngs(cfg["training"]["seed"]),
        in_channels=cfg["model"]["in_channels"],
        input_size=cfg["model"]["input_size"],
        enable_cond_image=cfg["model"]["enable_cond_image"],
        cond_channels=cfg["model"]["cond_channels"],
        num_cosmo_tokens=cfg["model"]["num_cosmo_tokens"],
        cond_patch_size=cfg["model"]["cond_patch_size"],
        cond_start=cfg["model"]["cond_start"],
        attn_drop=cfg["model"]["attn_drop"],
        proj_drop=cfg["model"]["proj_drop"],
        split_qkv=cfg["model"].get("split_qkv", False),
        mask_theta_to_field=cfg["model"].get("mask_theta_to_field", False),
    )
    model = Denoiser(model, cfg)
    nnx.update(model, states)
    model.t_eps = 0.05  # match plot_amortized.py
    print(f"Loaded model from {args.checkpoint} ({args.ckpt_tag})")

    # -- Simulator: same recipe as mcmc.py / plot_amortized.py, Planck15
    sim_kwargs = dict(
        N=128, map_size=5,
        gal_per_arcmin2=config_lsst_y_10.gals_per_arcmin2,
        sigma_e=config_lsst_y_10.sigma_e,
        nbins=config_lsst_y_10.nbins,
        a=config_lsst_y_10.a, b=config_lsst_y_10.b, z0=config_lsst_y_10.z0,
        model_type="lognormal", lognormal_shifts="LSSTY10",
    )
    cosmo = jc.parameters.Planck15()
    cond_dict = {
        "omega_c": cosmo.Omega_c, "omega_b": cosmo.Omega_b,
        "sigma_8": cosmo.sigma8, "h_0": cosmo.h,
        "n_s": cosmo.n_s, "w_0": cosmo.w0,
    }

    def draw(key, with_noise):
        m = partial(lensingLogNormal, with_noise=with_noise, **sim_kwargs)
        m = seed(m, key)
        m = condition(m, cond_dict)
        tr = trace(m).get_trace()
        return tr["y"]["value"]

    # Same key, with vs without noise: numpyro's per-site sub-keys mean the
    # field site sees the same sub-key in both traces, so the κ map is identical.
    sim_key = jax.random.key(args.seed)
    truth_kappa = np.asarray(draw(sim_key, with_noise=False))
    obs = np.asarray(draw(sim_key, with_noise=True))

    print(f"truth κ: {truth_kappa.shape}, obs: {obs.shape}")
    delta = obs - truth_kappa
    print(f"obs - truth κ: mean={float(delta.mean()):.4g}, "
          f"std (per-bin)={[float(delta[..., c].std()) for c in range(5)]}")

    # -- JADE inference on the noisy obs
    cond_arr = (obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)

    @jax.jit
    def sample_batch(key, cond_arr):
        if args.sampler == "sde":
            sampler = AdjointMatchingSDESampler(
                model=model, num_steps=num_steps, t_eps=0.05,
                noise_scale=args.noise_scale)
        else:
            sampler = HeunSampler(model=model, num_steps=num_steps)
        keys = jax.random.split(key, 3)
        x_0 = jax.random.normal(keys[0], shape=(args.batch_size, 128, 128, 5))
        cosmo_0 = jax.random.normal(keys[1], shape=(args.batch_size, 6))
        sk = jax.random.split(keys[2], args.batch_size)
        x_s, c_s = jax.vmap(sampler, in_axes=(0, 0, None, 0))(x_0, cosmo_0, cond_arr, sk)
        return x_s, c_s

    n_batches = max(1, (args.n_samples + args.batch_size - 1) // args.batch_size)
    sampler_key = jax.random.key(args.sampler_seed)
    x_samples, cosmo_samples = [], []
    for _ in tqdm(range(n_batches), desc="JADE sampling"):
        sampler_key, sk = jax.random.split(sampler_key)
        x_s, c_s = sample_batch(sk, cond_arr)
        x_samples.append(np.asarray(x_s))
        cosmo_samples.append(np.asarray(c_s))
    x_samples = np.concatenate(x_samples, axis=0) * FIELD_STD + FIELD_MEAN
    cosmo_samples = (np.concatenate(cosmo_samples, axis=0)
                     / cfg["loss"]["SCALE_COSMO"] * THETA_STD + THETA_MEAN)
    print(f"x_samples: {x_samples.shape}, cosmo_samples: {cosmo_samples.shape}")

    # -- Power spectrum (lenstools); same binning as plot_amortized.py
    map_size_deg = 5
    l_edges = np.linspace(500, 4608.0, 128)

    def fill_lower_diag(arr, nl):
        n = int(np.sqrt(len(arr) * 2)) + 1
        mask = np.arange(n)[:, None] > np.arange(n)
        out = np.zeros((n, n, nl))
        out[np.stack(mask, axis=1)] = arr
        return out.T

    def compute_ps(m1, m2):
        bins = [0, 1, 2, 3, 4]
        cross_list = []
        ell = None
        for i, j in itertools.combinations(bins, 2):
            ell, ps = ConvergenceMap(m1[:, :, i], angle=map_size_deg * u.deg).cross(
                ConvergenceMap(m2[:, :, j], angle=map_size_deg * u.deg),
                l_edges=l_edges,
            )
            cross_list.append(ps)
        ps_cross = fill_lower_diag(np.array(cross_list), 127)
        auto_list = []
        for i in bins:
            ell, ps = ConvergenceMap(m1[:, :, i], angle=map_size_deg * u.deg).cross(
                ConvergenceMap(m2[:, :, i], angle=map_size_deg * u.deg),
                l_edges=l_edges,
            )
            auto_list.append(ps)
        return ell, np.array(auto_list), ps_cross

    ell, ps_auto_truth, ps_cross_truth = compute_ps(truth_kappa, truth_kappa)

    n_ps = min(64, len(x_samples))
    ps_auto_samp, ps_cross_samp = [], []
    xtruth_samp = []  # per-bin truth_i x sample_i cross spectrum
    for s in tqdm(range(n_ps), desc="κ samples C_ℓ"):
        _, a_s, c_s = compute_ps(x_samples[s], x_samples[s])
        ps_auto_samp.append(a_s)
        ps_cross_samp.append(c_s)
        _, a_x, _ = compute_ps(truth_kappa, x_samples[s])  # truth x sample
        xtruth_samp.append(a_x)
    ps_auto_samp = np.array(ps_auto_samp)
    ps_cross_samp = np.array(ps_cross_samp)
    xtruth_samp = np.array(xtruth_samp)  # (n_ps, 5, n_ell)

    auto_mean, auto_std = ps_auto_samp.mean(0), ps_auto_samp.std(0)
    cross_mean, cross_std = ps_cross_samp.mean(0), ps_cross_samp.std(0)

    # -- Plot: lower-triangular 5×5 grid of auto + cross spectra
    fontsize_text, fontsize_ticks, fontsize_legend = 32, 20, 18
    tick_major, tick_minor, tick_w = 8, 4, 1.4

    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            if j > i:
                ax[i, j].axis("off")
                continue
            if i == j:
                ax[i, j].loglog(ell, ps_auto_truth[i], color="k",
                                label=r"Truth $\kappa$")
                ax[i, j].plot(ell, auto_mean[i], color="tab:blue",
                              label="JADE samples")
                ax[i, j].fill_between(ell, auto_mean[i] - auto_std[i],
                                      auto_mean[i] + auto_std[i],
                                      color="tab:blue", alpha=0.3)
            else:
                ax[i, j].loglog(ell, ps_cross_truth[:, i, j], color="k")
                ax[i, j].plot(ell, cross_mean[:, i, j], color="tab:blue")
                ax[i, j].fill_between(ell, cross_mean[:, i, j] - cross_std[:, i, j],
                                      cross_mean[:, i, j] + cross_std[:, i, j],
                                      color="tab:blue", alpha=0.3)
            ax[i, j].set_xscale("log")
            ax[i, j].set_yscale("log")
            ax[i, j].set_xlim(ell.min(), ell.max())
            ax[i, j].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
            ax[i, j].yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax[i, j].xaxis.set_minor_formatter(NullFormatter())
            ax[i, j].yaxis.set_minor_formatter(NullFormatter())
            ax[i, j].tick_params(which="major", length=tick_major, width=tick_w)
            ax[i, j].tick_params(which="minor", length=tick_minor, width=tick_w)
            ax[i, j].tick_params(axis="x", labelsize=fontsize_ticks,
                                 labelbottom=(i == 4))
            ax[i, j].tick_params(axis="y", labelsize=fontsize_ticks,
                                 labelleft=(j == 0))

    fig.supxlabel(r"$\ell$", fontsize=fontsize_text)
    fig.supylabel(r"$\mathcal{C}_\ell$", fontsize=fontsize_text, x=-0.02)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right",
               bbox_to_anchor=(0.88, 0.88), fontsize=fontsize_legend)

    out_pdf = os.path.join(args.save_dir, "power-spectra-truth-vs-jade.pdf")
    out_png = os.path.join(args.save_dir, "power-spectra-truth-vs-jade.png")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}\nwrote {out_png}")

    # -- Cross-correlation coefficient between the truth κ and the JADE samples,
    # per tomographic bin: r_ℓ = C_ℓ^{truth×sample} / sqrt(C_ℓ^truth C_ℓ^sample).
    # r_ℓ -> 1 where the obs constrains the field's phases (good reconstruction),
    # and falls off at small scales / high ℓ where the posterior is prior-driven.
    r_ell = xtruth_samp / np.sqrt(ps_auto_truth[None] * ps_auto_samp)  # (n_ps,5,n_ell)
    r_mean, r_std = r_ell.mean(0), r_ell.std(0)

    fig_r, ax_r = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)
    for i in range(5):
        ax_r[i].axhline(1.0, color="k", lw=1.2, ls=":")
        ax_r[i].plot(ell, r_mean[i], color="tab:blue", label=r"JADE $\times$ truth")
        ax_r[i].fill_between(ell, r_mean[i] - r_std[i], r_mean[i] + r_std[i],
                             color="tab:blue", alpha=0.3)
        ax_r[i].set_xscale("log")
        ax_r[i].set_xlim(ell.min(), ell.max())
        ax_r[i].set_ylim(0.0, 1.1)
        ax_r[i].set_title(f"Bin {i}", fontsize=fontsize_ticks)
        ax_r[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax_r[i].xaxis.set_minor_formatter(NullFormatter())
        ax_r[i].tick_params(which="major", length=tick_major, width=tick_w)
        ax_r[i].tick_params(which="minor", length=tick_minor, width=tick_w)
        ax_r[i].tick_params(axis="x", labelsize=fontsize_ticks)
        ax_r[i].tick_params(axis="y", labelsize=fontsize_ticks, labelleft=(i == 0))
    fig_r.supxlabel(r"$\ell$", fontsize=fontsize_text)
    fig_r.supylabel(r"$r_\ell$", fontsize=fontsize_text, x=0.05)
    handles, labels = ax_r[0].get_legend_handles_labels()
    fig_r.legend(handles, labels, loc="upper right",
                 bbox_to_anchor=(0.99, 0.99), fontsize=fontsize_legend)
    xc_pdf = os.path.join(args.save_dir, "cross-correlation-truth-vs-jade.pdf")
    xc_png = os.path.join(args.save_dir, "cross-correlation-truth-vs-jade.png")
    plt.savefig(xc_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(xc_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_r)
    print(f"wrote {xc_pdf}\nwrote {xc_png}")

    # -- Relative power-spectrum error: (C_ℓ^truth - C_ℓ^sample) / C_ℓ^truth.
    rel_auto = (ps_auto_truth[None] - ps_auto_samp) / ps_auto_truth[None]  # (n_ps,5,n_ell)
    rel_auto_mean, rel_auto_std = rel_auto.mean(0), rel_auto.std(0)
    _safe = np.where(ps_cross_truth[None] == 0, 1.0, ps_cross_truth[None])
    rel_cross = (ps_cross_truth[None] - ps_cross_samp) / _safe  # (n_ps,n_ell,5,5)
    rel_cross_mean, rel_cross_std = rel_cross.mean(0), rel_cross.std(0)

    fig_rel, ax_rel = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            if j > i:
                ax_rel[i, j].axis("off")
                continue
            if i == j:
                ax_rel[i, j].axhline(0, color="k", lw=1.0, label=r"Truth $\kappa$")
                ax_rel[i, j].plot(ell, rel_auto_mean[i], color="tab:blue",
                                  label="JADE sample")
                ax_rel[i, j].fill_between(ell, rel_auto_mean[i] - rel_auto_std[i],
                                          rel_auto_mean[i] + rel_auto_std[i],
                                          color="tab:blue", alpha=0.3)
            else:
                ax_rel[i, j].axhline(0, color="k", lw=1.0)
                ax_rel[i, j].plot(ell, rel_cross_mean[:, i, j], color="tab:blue")
                ax_rel[i, j].fill_between(ell, rel_cross_mean[:, i, j] - rel_cross_std[:, i, j],
                                          rel_cross_mean[:, i, j] + rel_cross_std[:, i, j],
                                          color="tab:blue", alpha=0.3)
            ax_rel[i, j].set_xscale("log")
            ax_rel[i, j].set_xlim(ell.min(), ell.max())
            ax_rel[i, j].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
            ax_rel[i, j].xaxis.set_minor_formatter(NullFormatter())
            ax_rel[i, j].tick_params(which="major", length=tick_major, width=tick_w)
            ax_rel[i, j].tick_params(which="minor", length=tick_minor, width=tick_w)
            ax_rel[i, j].tick_params(axis="x", labelsize=fontsize_ticks, labelbottom=(i == 4))
            ax_rel[i, j].tick_params(axis="y", labelsize=fontsize_ticks, labelleft=(j == 0))
    fig_rel.supxlabel(r"$\ell$", fontsize=fontsize_text)
    fig_rel.supylabel(
        r"$(\mathcal{C}_\ell^{\rm truth}-\mathcal{C}_\ell^{\rm sample})/\mathcal{C}_\ell^{\rm truth}$",
        fontsize=fontsize_text * 0.7, x=-0.02)
    handles, labels = ax_rel[0, 0].get_legend_handles_labels()
    fig_rel.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(0.88, 0.88), fontsize=fontsize_legend)
    rel_pdf = os.path.join(args.save_dir, "power-spectra-relative-truth-vs-jade.pdf")
    rel_png = os.path.join(args.save_dir, "power-spectra-relative-truth-vs-jade.png")
    plt.savefig(rel_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(rel_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_rel)
    print(f"wrote {rel_pdf}\nwrote {rel_png}")

    # -- % error summary (auto bins): mean over ℓ of |C_truth - <C_sample>| / C_truth.
    err_auto = np.abs(ps_auto_truth - auto_mean) / ps_auto_truth  # (5, n_ell)
    per_bin_pct = err_auto.mean(axis=1) * 100.0                    # (5,)
    overall_pct = float(per_bin_pct.mean())
    summary = ["Relative power-spectrum error (auto, |ΔC/C| averaged over ℓ):"]
    summary += [f"  bin {i}: {per_bin_pct[i]:.2f} %" for i in range(5)]
    summary += [f"  averaged across bins: {overall_pct:.2f} %"]
    summary = "\n".join(summary)
    print(summary)
    with open(os.path.join(args.save_dir, "relative_error_summary.txt"), "w") as fh:
        fh.write(summary + "\n")

    np.savez(
        os.path.join(args.save_dir, "truth_vs_jade_ps.npz"),
        truth_kappa=truth_kappa, obs=obs,
        x_samples=x_samples, cosmo_samples=cosmo_samples,
        ell=ell, ps_auto_truth=ps_auto_truth, ps_cross_truth=ps_cross_truth,
        ps_auto_samples=ps_auto_samp, ps_cross_samples=ps_cross_samp,
        xtruth_samples=xtruth_samp, r_ell=r_ell,
        rel_auto_mean=rel_auto_mean, per_bin_pct=per_bin_pct, overall_pct=overall_pct,
    )
    print(f"wrote {os.path.join(args.save_dir, 'truth_vs_jade_ps.npz')}")


if __name__ == "__main__":
    main()
