"""Per-noise-level (per-t) field v-loss diagnostic, 1M vs 100k.

Replicates jade.flow.FlowLoss exactly (v-loss: v=(x-x_pred)/clip(1-t,0.05)),
but evaluates at FIXED t on a grid instead of sampling t. For each t we report:
  - field v-loss  E[(v - v_pred)^2]
  - target power  E[v^2]            (= loss of the predict-zero-velocity baseline)
  - R^2 = 1 - loss/power            (fit quality, scale-normalised, in [.,1])
overlaid with the TRAINING time density p(t) = sigmoid((z+mu)*sigma).

Reading it: low R^2 at some t = score poorly fit there. If that coincides with
low p(t) at the spread-setting (higher-noise / small-t) end, that's the
weighting/time-distribution hypothesis (lever 1). NOTE: v-loss/score-fit is an
INDIRECT proxy for calibration -- this is hypothesis-generating, not a verdict.
"""
import os
import argparse

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.unicode_minus": False})

from jade.nn_hybrid import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, sigma_lsst
from jade.flow import Denoiser
from jade.utils import load_model
from datasets import load_from_disk

MODELS = {
    "million": dict(ckpt="/u/bremy/repos/jade/experiments/wandb/run-20260615_153845-fk49rnft/files/checkpoints",
                    state="JADE_B_16_ema_best"),
    "former":  dict(ckpt="/u/bremy/repos/jade/experiments/wandb/run-20260507_170014-7hnur00g/files/checkpoints",
                    state="JADE_B_16_latest"),
}
# Common held-out evaluation data (same simulator for both models).
DATASET = "/work/hdd/benb/bremy/sbi_lens_million_full"
VAL_SPLIT, SHUFFLE_SEED = 0.05, 42


def build_model(spec):
    cfg, states = load_model(spec["ckpt"], spec["state"])
    m = JADE_B_16(
        rngs=nnx.Rngs(cfg['training']['seed']),
        in_channels=cfg['model']['in_channels'], input_size=cfg['model']['input_size'],
        enable_cond_image=cfg['model']['enable_cond_image'], cond_channels=cfg['model']['cond_channels'],
        num_cosmo_tokens=cfg['model']['num_cosmo_tokens'], cond_patch_size=cfg['model']['cond_patch_size'],
        cond_start=cfg['model']['cond_start'], attn_drop=cfg['model']['attn_drop'],
        proj_drop=cfg['model']['proj_drop'],
        split_qkv=cfg['model'].get('split_qkv', False),
        mask_theta_to_field=cfg['model'].get('mask_theta_to_field', False),
    )
    m = Denoiser(m, cfg)
    nnx.update(m, states)
    return m, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n-noise", type=int, default=4, help="noise draws averaged per t")
    p.add_argument("--n-t", type=int, default=41)
    p.add_argument("--output", default="tarp_results/diag_pert_field_loss.pdf")
    args = p.parse_args()

    # --- common eval batch (normalized field + cosmo + cond) ---
    ds = load_from_disk(DATASET).train_test_split(
        test_size=VAL_SPLIT, seed=SHUFFLE_SEED)["test"].with_format("numpy")
    b = ds[:args.batch]
    x = (np.asarray(b["map"]) - np.asarray(FIELD_MEAN).reshape(1,1,1,-1)) / np.asarray(FIELD_STD).reshape(1,1,1,-1)
    cosmo = (np.asarray(b["theta"]) - np.asarray(THETA_MEAN)) / np.asarray(THETA_STD)
    x = jnp.asarray(x, jnp.float32); cosmo = jnp.asarray(cosmo, jnp.float32)
    ckey = jax.random.PRNGKey(0)
    raw = x * FIELD_STD.reshape(1,1,1,-1) + FIELD_MEAN.reshape(1,1,1,-1)
    cond = raw + sigma_lsst.reshape((1,1,1,-1)) * jax.random.normal(ckey, shape=x.shape)
    cond = (cond - FIELD_MEAN.reshape(1,1,1,-1)) / FIELD_STD.reshape(1,1,1,-1)

    ts = np.linspace(0.02, 0.98, args.n_t)
    dummy = jnp.zeros((args.batch, 2), dtype=jnp.uint32)

    def per_t_for_model(model):
        @jax.jit
        def eval_t(t_scalar, key):
            t = jnp.full((args.batch,), t_scalar)
            def one(key):
                nx = jax.random.normal(key, shape=x.shape)
                nc = jax.random.normal(jax.random.fold_in(key, 1), shape=cosmo.shape)
                xt = t[:,None,None,None]*x + (1-t)[:,None,None,None]*nx
                ct = t[:,None]*cosmo + (1-t)[:,None]*nc
                xp, _ = jax.vmap(model.x_pred, in_axes=(0,0,0,0,None,0))(xt, ct, t, cond, False, dummy)
                c = jnp.clip(1 - t[:,None,None,None], a_min=0.05)
                v = (x - xt)/c; vp = (xp - xt)/c
                return jnp.mean((v-vp)**2), jnp.mean(v**2)
            ls, pw = jax.vmap(one)(jax.random.split(key, args.n_noise))
            return ls.mean(), pw.mean()
        loss, power = [], []
        for i, tv in enumerate(ts):
            l, pw = eval_t(float(tv), jax.random.PRNGKey(1000+i))
            loss.append(float(l)); power.append(float(pw))
        return np.array(loss), np.array(power)

    results = {}
    for name, spec in MODELS.items():
        print(f"evaluating {name} ({spec['state']}) ...", flush=True)
        m, cfg = build_model(spec)
        loss, power = per_t_for_model(m)
        r2 = 1 - loss/power
        results[name] = (loss, power, r2)
        print(f"  {name}: R^2 min={r2.min():.3f} @ t={ts[r2.argmin()]:.2f}; "
              f"R^2@t=0.1={np.interp(0.1,ts,r2):.3f}, @t=0.5={np.interp(0.5,ts,r2):.3f}")
        del m

    # training time density p(t)
    z = np.random.RandomState(0).randn(200000)
    t_samp = 1/(1+np.exp(-((z - 0.8)*0.8)))   # sigmoid((z+mu)*sigma), mu=-0.8

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax in (a1, a2):
        axt = ax.twinx(); axt.hist(t_samp, bins=60, range=(0,1), density=True,
                                   color="0.85", zorder=0); axt.set_yticks([])
        axt.set_ylabel("training p(t)", color="0.6")
    for name, c in [("million","tab:blue"), ("former","tab:orange")]:
        loss, power, r2 = results[name]
        a1.plot(ts, loss, color=c, label=f"{name} v-loss")
        a2.plot(ts, r2, color=c, label=f"{name} $R^2$")
    a1.plot(ts, results["million"][1], "k--", lw=1, label="target power E[v²]")
    a1.set_yscale("log"); a1.set_xlabel("t  (0=noise, 1=data)"); a1.set_ylabel("field v-loss"); a1.legend(fontsize=8)
    a2.set_xlabel("t  (0=noise, 1=data)"); a2.set_ylabel("$R^2$ (fit quality)")
    a2.set_ylim(0,1); a2.legend(fontsize=8)
    fig.suptitle("Per-t field v-loss: 1M vs 100k (shaded = training time density)")
    fig.tight_layout()
    fig.savefig(args.output); fig.savefig(os.path.splitext(args.output)[0]+".png", dpi=200)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
