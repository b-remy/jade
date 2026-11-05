#!/usr/bin/env python
"""Train JADE as an *unconditional* joint model of (cosmology, convergence map).

train.py trains the conditional denoiser: a noisy shear field is embedded by
`cond_embedder` and injected into the token sequence at `cond_start`, so the
sampler returns p(theta, kappa | gamma_obs). This script trains the same
architecture with the conditioning branch switched off, giving the joint prior

    p(theta, kappa)

which the sampler draws from without any observation. Differences from train.py:

  * `model.enable_cond_image` is forced to False and `cond=None` is passed
    everywhere, so no shape-noise conditioning field is ever built. The
    `cond_embedder` and the conditioned RoPE table are not created at all.
  * The per-epoch corner plot against the reference MCMC chain is gone (there is
    no observation to condition on) and is replaced by a prior-recovery corner:
    the model's marginal p(theta) against the held-out simulation thetas. That
    is the diagnostic that matters here — an unconditional joint model should
    reproduce the training prior exactly.
  * A conditional checkpoint can still be used as a warm start: `cond_embedder`
    is the only parameter group the unconditional model does not have, so it is
    stripped from the loaded state (see `drop_cond_params`).
  * The EMA/live parameter swap around validation copies the live parameters
    instead of aliasing them, so the `latest`/`best` tags really hold live
    weights here. train.py keeps the aliasing on purpose, to stay faithful to
    how the published checkpoints were produced.
"""

import multiprocessing
import os

# DataLoader workers (spawn/forkserver) re-import this module; keep them off GPU
# so they don't each try to grab a CUDA context and OOM.
if multiprocessing.current_process().name != "MainProcess":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse

import dm_pix as pix
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
import yaml
from datasets import load_from_disk
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.utils.data import DataLoader
from tqdm import tqdm

from jade.flow import Denoiser, FlowLoss
from jade.init import FIELD_MEAN, FIELD_STD, THETA_MEAN, THETA_STD
from jade.nn import JADE_B_16
from jade.sampling import HeunSampler
from jade.utils import dump_model, load_config, load_model, plot_denoiser, plot_samples

PARAM_LABELS = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]


@jax.jit
@jax.vmap
def augment(x, key):
    keys = jax.random.split(key, 2)
    x = pix.random_flip_left_right(keys[0], x)
    x = pix.random_flip_up_down(keys[1], x)
    return x


def hf_collate(batch):
    """Stack the two columns training needs, leaving the rest unread."""
    return {k: np.stack([b[k] for b in batch]) for k in ("map", "theta")}


@jax.jit
def normalize_batch(batch):
    return {
        "map": (batch["map"] - FIELD_MEAN.reshape(1, 1, 1, -1)) / FIELD_STD.reshape(1, 1, 1, -1),
        "theta": (batch["theta"] - THETA_MEAN) / THETA_STD,
    }


def shard_state(state, mesh):
    """Shard each array over its first divisible axis, replicating the rest.

    Axis order matters: conv kernels are (kH, kW, in, out) and want splitting
    over output channels rather than over the kernel window.
    """
    num_devices = mesh.shape["fsdp"]
    preference = {1: (0,), 2: (0, 1), 4: (3, 2)}

    def sharding(x):
        if x.ndim == 0 or x.size == 0:
            return NamedSharding(mesh, P())
        for axis in preference.get(x.ndim, range(x.ndim)):
            if x.shape[axis] % num_devices == 0:
                if x.ndim == 2 and axis == 0:
                    # Row-sharded matrix: the trailing replicated axis is
                    # implicit, so this is the plain P('fsdp').
                    return NamedSharding(mesh, P("fsdp"))
                spec = [None] * x.ndim
                spec[axis] = "fsdp"
                return NamedSharding(mesh, P(*spec))
        return NamedSharding(mesh, P())

    return jax.tree.map(lambda x: jax.device_put(x, sharding(x)), state)


def create_optimizer(cfg, total_steps):
    opt = cfg["optimizer"]
    if opt["use_schedule"]:
        lr = optax.linear_schedule(
            init_value=opt["schedule"]["init_value"],
            end_value=opt["schedule"]["end_value"],
            transition_steps=total_steps,
        )
    else:
        lr = opt["learning_rate"]
    return optax.chain(
        optax.clip_by_global_norm(opt["grad_clip_norm"]),
        optax.adamw(learning_rate=lr, b1=opt["beta1"], b2=opt["beta2"], weight_decay=opt["weight_decay"]),
    )


def make_loader(ds, cfg, shuffle):
    workers = cfg["data"].get("num_workers", 8)
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=hf_collate,
        drop_last=shuffle,
        persistent_workers=workers > 0,
        pin_memory=False,
        # Batches prefetched per worker: more reads in flight to hide the
        # filesystem's small-random-read latency.
        prefetch_factor=cfg["data"].get("prefetch_factor", 4) if workers else None,
        multiprocessing_context="forkserver" if workers else None,
    )


def drop_cond_params(state):
    """Strip the conditioning embedder from a checkpoint of a conditional run.

    `cond_embedder` is the only parameter group a conditional JADE has and an
    unconditional one does not (the conditioned RoPE tables are plain arrays,
    not parameters), so removing it makes a stage-1/stage-2 checkpoint a valid
    warm start for this script. Everything else — patch embedder, cosmology
    embedder, all 12 blocks, both heads — transfers as is.
    """
    params = state["model"] if "model" in state else state
    if "cond_embedder" in params:
        del params["cond_embedder"]
        print("warm start: dropped cond_embedder (conditional checkpoint -> unconditional model)")
    return state


def sample_joint(model, cfg, key, n, chunk=64):
    """Draw n unconditional joint samples (field, cosmology). Returns (key, x, cosmo)."""
    size, channels = cfg["model"]["input_size"], cfg["model"]["in_channels"]
    cosmo_dim = cfg["model"].get("cosmo_dim", 6)
    sampler = HeunSampler(model=model, num_steps=cfg.get("sampling", {}).get("num_steps", 50))

    fields, cosmos = [], []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        key, key_x, key_c, key_s = jax.random.split(key, 4)
        x_0 = jax.random.normal(key_x, shape=(m, size, size, channels))
        cosmo_0 = jax.random.normal(key_c, shape=(m, cosmo_dim))
        x_s, cosmo_s = jax.vmap(sampler, in_axes=(0, 0, None, 0))(x_0, cosmo_0, None, jax.random.split(key_s, m))
        fields.append(np.asarray(x_s))
        cosmos.append(np.asarray(cosmo_s))

    return key, np.concatenate(fields), np.concatenate(cosmos)


def plot_prior_recovery(theta_model, theta_data):
    """Model marginal p(theta) against the held-out simulation thetas.

    Both arrays are in physical units. A well trained unconditional model puts
    the two sets of contours on top of each other; a systematic offset or a
    collapsed width is the failure mode to watch for.
    """
    from getdist import MCSamples, plots

    s_model = MCSamples(samples=np.asarray(theta_model), names=PARAM_LABELS, label="Model")
    s_data = MCSamples(samples=np.asarray(theta_data), names=PARAM_LABELS, label="Simulations")

    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 14
    g.settings.axes_labelsize = 16
    g.settings.legend_fontsize = 16
    g.triangle_plot(
        [s_model, s_data],
        PARAM_LABELS,
        filled=[True, False],
        contour_colors=["#d06e99ff", "black"],
        contour_ls=["-", "--"],
        contour_lws=[2.0, 2.0],
    )
    return plt.gcf()


def log_visuals(cfg, model, key, x_val, cosmo_val, theta_data):
    """Denoiser panel, joint samples and the prior-recovery corner. Returns the key."""
    scale = cfg["loss"]["SCALE_COSMO"]

    # plot_denoiser builds a noisy conditioning field unconditionally; the model
    # ignores it because enable_cond_image is False.
    fig = plot_denoiser(x_val, cosmo_val, model, key, cfg)
    wandb.log({"denoiser_ema" if cfg["ema"]["use_ema"] else "denoiser": wandb.Image(fig)})
    plt.close(fig)

    n = cfg["logging"].get("num_prior_samples", 256)
    key, subkey = jax.random.split(key, 2)
    key, x_samples, cosmo_samples = sample_joint(model, cfg, subkey, n)

    fig = plot_samples(x_samples[:6], cosmo_samples[:6] / scale, n_samples=6)
    wandb.log({"samples": wandb.Image(fig)})
    plt.close(fig)

    # getdist's KDE needs a non-trivial number of draws; below that it raises
    # rather than returning a thin contour, so skip instead of killing the run.
    if theta_data is not None and n >= 64:
        theta_model = cosmo_samples / scale * THETA_STD + THETA_MEAN
        wandb.log({"prior_recovery": wandb.Image(plot_prior_recovery(theta_model, theta_data))})
        plt.close("all")

    return key


def save_checkpoints(cfg, model, ema_params, checkpoint_dir, best):
    name = cfg["model"]["name"]
    live = nnx.state(model, nnx.Param)
    for suffix, state in [("ema_latest", ema_params), ("latest", live)]:
        if state is not None:
            dump_model(cfg, state, f"{name}_{suffix}", checkpoint_dir)
    if best:
        for suffix, state in [("ema_best", ema_params), ("best", live)]:
            if state is not None:
                dump_model(cfg, state, f"{name}_{suffix}", checkpoint_dir)


def train(cfg):
    # Force the unconditional architecture before the config is logged, so the
    # run record matches what was actually built.
    if cfg["model"].get("enable_cond_image", False):
        print("model.enable_cond_image=true in the config; overriding to false (unconditional run)")
    cfg["model"]["enable_cond_image"] = False

    num_devices = jax.device_count()
    use_sharding = cfg.get("distributed", {}).get("use_sharding", False) and num_devices > 1
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=("fsdp",)) if use_sharding else None
    print(f"{num_devices} device(s) on {jax.devices()[0].platform}, FSDP {'on' if use_sharding else 'off'}")

    if cfg["training"]["use_mixed_precision"]:
        jax.config.update("jax_default_matmul_precision", "bfloat16")

    run = wandb.init(project=cfg["logging"]["project"], entity=cfg["logging"]["entity"], config=cfg)
    with open(os.path.join(wandb.run.dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    checkpoint_dir = os.path.abspath(os.path.join(wandb.run.dir, "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # JADE_DATASET_PATH lets the launcher point at a copy staged on node-local
    # NVMe without editing the config.
    data_path = os.environ.get("JADE_DATASET_PATH", cfg["data"]["dataset_path"])
    dataset = load_from_disk(data_path)
    dataset = dataset.select_columns([c for c in ("map", "theta") if c in dataset.column_names])
    dataset = dataset.with_format("numpy").train_test_split(
        test_size=cfg["data"]["val_split"], seed=cfg["data"]["shuffle_seed"]
    )
    ds_train, ds_val = dataset["train"], dataset["test"]
    print(f"{data_path}: {len(ds_train)} train, {len(ds_val)} val")

    # Reference marginal for the prior-recovery corner: held-out thetas in
    # physical units. One columnar read, so the maps are not touched.
    n_ref = min(len(ds_val), cfg["logging"].get("num_reference_thetas", 5000))
    theta_data = np.asarray(ds_val.select(range(n_ref))["theta"])

    model = Denoiser(
        JADE_B_16(
            rngs=nnx.Rngs(cfg["training"]["seed"]),
            in_channels=cfg["model"]["in_channels"],
            input_size=cfg["model"]["input_size"],
            patch_size=cfg["model"]["patch_size"],
            enable_cond_image=False,
            num_cosmo_tokens=cfg["model"]["num_cosmo_tokens"],
            attn_drop=cfg["model"]["attn_drop"],
            proj_drop=cfg["model"]["proj_drop"],
            attn_impl=cfg["model"].get("attn_impl", "dense"),
        ),
        cfg,
    )

    # Finetuning seeds the EMA with the loaded weights, so the EMA is live from
    # the first step; training from scratch initializes it after one epoch.
    ema_params = None
    if cfg["start_from_checkpoint"]:
        tag = f"{cfg['model']['name']}_{cfg.get('params_tag', 'ema_latest')}"
        print(f"loading {tag} from {cfg['params_path']}")
        _, ema_params = load_model(cfg["params_path"], tag)
        ema_params = drop_cond_params(ema_params)
        if mesh is not None:
            ema_params = shard_state(ema_params, mesh)
        nnx.update(model, ema_params)

    if mesh is not None:
        nnx.update(model, shard_state(nnx.state(model, nnx.Param), mesh))

    steps_per_epoch = len(ds_train) // cfg["training"]["batch_size"]
    total_steps = cfg["training"]["num_epochs"] * steps_per_epoch
    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"{n_params:,} parameters, {total_steps} steps")

    optimizer = nnx.Optimizer(model, create_optimizer(cfg, total_steps), wrt=nnx.Param)
    loss_fn = FlowLoss(cfg)

    @nnx.jit
    def train_step(model, optimizer, x, cosmo, key):
        (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model=model, x=x, cosmo=cosmo, cond=None, key=key, lambda_cosmo=cfg["loss"]["lambda_cosmo"], train=True
        )
        optimizer.update(model, grads)
        return loss

    @jax.jit
    def update_ema(ema, params, decay):
        return jax.tree.map(lambda e, p: decay * e + (1 - decay) * p, ema, params)

    scale = cfg["loss"]["SCALE_COSMO"]
    use_ema = cfg["ema"]["use_ema"]
    num_epochs = cfg["training"]["num_epochs"]
    val_max_batches = cfg.get("validation", {}).get("max_batches", None)

    train_loader = make_loader(ds_train, cfg, shuffle=True)
    val_loader = make_loader(ds_val, cfg, shuffle=False)

    key = jax.random.key(cfg["training"]["seed"])
    best_cosmo_loss = float("inf")
    step = 0

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{num_epochs}"):
            batch = normalize_batch(batch)

            key, subkey = jax.random.split(key, 2)
            x = augment(batch["map"], jax.random.split(subkey, len(batch["map"])))
            cosmo = batch["theta"] * scale

            key, subkey = jax.random.split(key, 2)
            loss = train_step(model, optimizer, x, cosmo, subkey)

            if use_ema and ema_params is None and epoch >= 1 and not cfg["start_from_checkpoint"]:
                ema_params = jax.tree.map(lambda a: a.copy(), nnx.state(model, nnx.Param))
                if mesh is not None:
                    ema_params = shard_state(ema_params, mesh)
                print(f"EMA initialized at epoch {epoch}")
            if use_ema and ema_params is not None:
                ema_params = update_ema(ema_params, nnx.state(model, nnx.Param), cfg["ema"]["decay"])

            if step % cfg["logging"]["log_every_n_steps"] == 0:
                run.log({"train/loss_total": loss, "train/epoch": epoch})
            step += 1

        # Validate, visualize and checkpoint on the EMA weights when they exist.
        # nnx.state returns a view, so the live parameters have to be copied out
        # before the swap or they would follow it.
        if use_ema and ema_params is not None:
            live_params = jax.tree.map(lambda a: a.copy(), nnx.state(model, nnx.Param))
            nnx.update(model, ema_params)

        losses = []
        for i, batch in enumerate(val_loader):
            if val_max_batches is not None and i >= val_max_batches:
                break
            batch = normalize_batch(batch)
            x_val, cosmo_val = batch["map"], batch["theta"] * scale
            key, val_key = jax.random.split(key, 2)
            losses.append(
                loss_fn(
                    model=model,
                    x=x_val,
                    cosmo=cosmo_val,
                    key=val_key,
                    lambda_cosmo=cfg["loss"]["lambda_cosmo"],
                    train=False,
                    cond=None,
                )
            )

        val_loss = np.mean([l[0] for l in losses])
        val_loss_x = np.mean([l[1][0] for l in losses])
        val_loss_cosmo = np.mean([l[1][1] for l in losses])
        run.log(
            {
                "val/loss_total": val_loss,
                "val/v_loss_field": val_loss_x,
                "val/v_loss_cosmo": val_loss_cosmo,
                "epoch": epoch + 1,
            }
        )
        print(f"epoch {epoch + 1}: val {val_loss:.4f} (field {val_loss_x:.4f}, cosmo {val_loss_cosmo:.4f})")

        if (epoch + 1) % cfg["logging"]["visualize_every_n_epochs"] == 0:
            key = log_visuals(cfg, model, key, x_val, cosmo_val, theta_data)

        if use_ema and ema_params is not None:
            nnx.update(model, live_params)

        if (epoch + 1) % cfg["checkpoint"]["save_every_n_epochs"] == 0:
            # Best is tracked on the cosmology loss, not the total.
            best = cfg["checkpoint"]["keep_best"] and val_loss_cosmo < best_cosmo_loss
            if best:
                best_cosmo_loss = val_loss_cosmo
            save_checkpoints(cfg, model, ema_params if use_ema else None, checkpoint_dir, best)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the unconditional joint JADE model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file, e.g. configs/unconditional.yaml",
    )
    train(load_config(parser.parse_args().config))
