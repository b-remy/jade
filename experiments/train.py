#!/usr/bin/env python

import multiprocessing
import os

# DataLoader workers (spawn/forkserver) re-import this module; keep them off GPU
# so they don't each try to grab a CUDA context and OOM.
if multiprocessing.current_process().name != "MainProcess":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import pickle

import dm_pix as pix
import jax
import jax.numpy as jnp
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
from jade.init import FIELD_MEAN, FIELD_STD, THETA_MEAN, THETA_STD, sigma_lsst
from jade.nn import JADE_B_16
from jade.paths import MCMC_REF_DIR
from jade.sampling import HeunSampler
from jade.utils import dump_model, load_config, load_model, plot_corner, plot_denoiser, plot_samples


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


@jax.jit
def make_cond(x, key):
    """Add LSST shape noise to a normalized field, then renormalize."""
    mean, std = FIELD_MEAN.reshape(1, 1, 1, -1), FIELD_STD.reshape(1, 1, 1, -1)
    noisy = x * std + mean + sigma_lsst.reshape((1, 1, 1, -1)) * jax.random.normal(key, shape=x.shape)
    return (noisy - mean) / std


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


def load_corner_reference(cfg):
    """Fixed observation and reference chain for the per-epoch corner plot."""
    ref_dir = cfg.get("logging", {}).get("mcmc_ref_dir", str(MCMC_REF_DIR))
    try:
        with open(os.path.join(ref_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
            ref = pickle.load(f)
        with open(os.path.join(ref_dir, "mcmc_log_posterior_samples.pkl"), "rb") as f:
            chain = pickle.load(f)
    except FileNotFoundError as e:
        print(f"no corner reference ({e}); skipping corner plots")
        return None
    return jnp.asarray(ref["y"]), chain, np.asarray(ref["theta"])


def log_visuals(cfg, model, key, x_val, cosmo_val, corner):
    """Denoiser panel, unconditional samples and corner plot. Returns the key."""
    scale = cfg["loss"]["SCALE_COSMO"]
    use_cond = cfg["model"]["enable_cond_image"]

    fig = plot_denoiser(x_val, cosmo_val, model, key, cfg)
    wandb.log({"denoiser_ema" if cfg["ema"]["use_ema"] else "denoiser": wandb.Image(fig)})
    plt.close(fig)

    key, subkey = jax.random.split(key, 2)
    sampler = HeunSampler(model=model, num_steps=50)

    keys = jax.random.split(subkey, 3)
    x_0 = jax.random.normal(keys[0], shape=(6, 128, 128, 5))
    cosmo_0 = jax.random.normal(keys[1], shape=(6, 6))

    keys = jax.random.split(keys[2], 6)
    cond = make_cond(x_val[:6], keys[2]) if use_cond else None
    x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, cond, keys)

    fig = plot_samples(x_samples, cosmo_samples / scale, n_samples=6)
    wandb.log({"samples": wandb.Image(fig)})
    plt.close(fig)

    if corner is not None and use_cond:
        corner_obs, corner_mcmc, corner_truth = corner
        key, subkey = jax.random.split(key, 2)
        n = 128
        keys = jax.random.split(subkey, 3)
        x_0 = jax.random.normal(keys[0], shape=(n, 128, 128, 5))
        cosmo_0 = jax.random.normal(keys[1], shape=(n, 6))
        cond = (corner_obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)
        _, cosmo_post = jax.vmap(sampler, in_axes=(0, 0, None, 0))(x_0, cosmo_0, cond, jax.random.split(keys[2], n))
        theta_post = np.asarray(cosmo_post) / scale * THETA_STD + THETA_MEAN
        wandb.log({"corner": wandb.Image(plot_corner(theta_post, corner_mcmc, corner_truth))})
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

    model = Denoiser(
        JADE_B_16(
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
    def train_step(model, optimizer, x, cosmo, key, cond=None):
        (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model=model, x=x, cosmo=cosmo, cond=cond, key=key, lambda_cosmo=cfg["loss"]["lambda_cosmo"], train=True
        )
        optimizer.update(model, grads)
        return loss

    @jax.jit
    def update_ema(ema, params, decay):
        return jax.tree.map(lambda e, p: decay * e + (1 - decay) * p, ema, params)

    scale = cfg["loss"]["SCALE_COSMO"]
    use_cond = cfg["model"]["enable_cond_image"]
    use_ema = cfg["ema"]["use_ema"]
    num_epochs = cfg["training"]["num_epochs"]
    val_max_batches = cfg.get("validation", {}).get("max_batches", None)

    corner = load_corner_reference(cfg)
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
            cond = make_cond(x, subkey) if use_cond else None

            key, subkey = jax.random.split(key, 2)
            loss = train_step(model, optimizer, x, cosmo, subkey, cond=cond)

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
        if use_ema and ema_params is not None:
            live_params = nnx.state(model, nnx.Param)
            nnx.update(model, ema_params)

        losses = []
        for i, batch in enumerate(val_loader):
            if val_max_batches is not None and i >= val_max_batches:
                break
            batch = normalize_batch(batch)
            x_val, cosmo_val = batch["map"], batch["theta"] * scale
            key, val_key = jax.random.split(key, 2)
            cond_val = make_cond(x_val, val_key) if use_cond else None
            losses.append(
                loss_fn(
                    model=model,
                    x=x_val,
                    cosmo=cosmo_val,
                    key=val_key,
                    lambda_cosmo=cfg["loss"]["lambda_cosmo"],
                    train=False,
                    cond=cond_val,
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
            key = log_visuals(cfg, model, key, x_val, cosmo_val, corner)

        # nnx.state returns a view, so `live_params` tracks the swap above and
        # this restores the EMA weights rather than the live ones. Every saved
        # tag therefore holds EMA weights; the non-EMA tags are aliases. Kept
        # as-is because the published checkpoints were produced this way.
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
    parser = argparse.ArgumentParser(description="Train the JADE model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file, e.g. configs/hybrid.yaml (stage 1) or "
        "configs/finetune.yaml (stage 2, the paper model)",
    )
    train(load_config(parser.parse_args().config))
