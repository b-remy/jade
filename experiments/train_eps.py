#!/usr/bin/env python
"""Fine-tune the unconditional joint model into an Exact Posterior Score (EPS)
denoiser for the shear likelihood (arXiv:2606.17048, Algorithm 1).

DPS and MMPS (`sample_posterior_guidance.py`) use the unconditional prior as is,
and approximate the likelihood at sampling time. EPS does the opposite: the
likelihood enters in closed form through the posterior pivot

    mu_star(x_t, gamma, t) = Sigma_star (alpha_t / sigma_t^2 x_t + A^T Sigma_y^{-1} gamma)

and the network is fine-tuned to denoise that pivot (see `jade.flow_eps`).
Sampling then needs one plain network evaluation per step, with no VJP and no
linear solve (`sample_posterior_eps.py`).

Differences from train_unconditional.py:

  * The model is wrapped in `EPSDenoiser` and the loss is `EPSLoss`, which
    draws one noisy shear observation `A kappa + n`, `n ~ N(0, sigma_lsst^2)`,
    per training map. Same forward operator and noise as DPS/MMPS, so all three
    target the same posterior.
  * A warm start is required: the run starts from the unconditional prior (the
    paper fine-tunes rather than retraining, and the pivot equals x_t on every
    unobserved direction, so the prior is already the right model there).
  * The per-epoch diagnostic is the corner plot against the reference NUTS
    chain, on the same observation as `sample_posterior_guidance.py`, instead
    of the prior-recovery corner.
  * Checkpoints hold the inner `Denoiser` state, so they have exactly the
    layout of an unconditional checkpoint; `cfg["eps"]` marks them as EPS
    weights for the sampling script.
"""

import multiprocessing
import os

# DataLoader workers (spawn/forkserver) re-import this module; keep them off GPU
# so they don't each try to grab a CUDA context and OOM.
if multiprocessing.current_process().name != "MainProcess":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse

import jax
import matplotlib.pyplot as plt
import numpy as np
import wandb
import yaml
from datasets import load_from_disk
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from sample_posterior_guidance import shear_observation
from tqdm import tqdm
from train import load_corner_reference
from train_unconditional import (
    augment,
    create_optimizer,
    drop_cond_params,
    make_loader,
    normalize_batch,
    save_checkpoints,
    shard_state,
)

from jade.flow import Denoiser
from jade.flow_eps import EPSDenoiser, EPSLoss
from jade.init import THETA_MEAN, THETA_STD, sigma_lsst
from jade.nn import JADE_B_16
from jade.sampling import HeunSampler
from jade.utils import load_config, load_model, plot_corner


def log_corner(cfg, model, key, corner, n=128):
    """EPS posterior draws on the reference observation against the NUTS chain."""
    obs, chain, truth = corner
    gamma = shear_observation(obs)
    sampler = HeunSampler(model=model, num_steps=cfg.get("sampling", {}).get("num_steps", 50))

    keys = jax.random.split(key, 3)
    x_0 = jax.random.normal(keys[0], shape=(n, cfg["model"]["input_size"], cfg["model"]["input_size"], 5))
    cosmo_0 = jax.random.normal(keys[1], shape=(n, 6))
    _, cosmo_post = jax.vmap(sampler, in_axes=(0, 0, None, 0))(x_0, cosmo_0, gamma, jax.random.split(keys[2], n))

    theta_post = np.asarray(cosmo_post) / cfg["loss"]["SCALE_COSMO"] * THETA_STD + THETA_MEAN
    wandb.log({"corner": wandb.Image(plot_corner(theta_post, chain, truth))})
    plt.close("all")


def train(cfg):
    cfg["model"]["enable_cond_image"] = False
    if not cfg.get("start_from_checkpoint", False):
        raise ValueError("EPS fine-tunes the unconditional prior: set start_from_checkpoint and params_path.")
    # Read back by sample_posterior_eps.py, which refuses checkpoints without it.
    cfg["eps"] = {"likelihood": "shear", "operator": "ks93", "sigma_gamma": "sigma_lsst"}

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

    data_path = os.environ.get("JADE_DATASET_PATH", cfg["data"]["dataset_path"])
    dataset = load_from_disk(data_path)
    dataset = dataset.select_columns([c for c in ("map", "theta") if c in dataset.column_names])
    dataset = dataset.with_format("numpy").train_test_split(
        test_size=cfg["data"]["val_split"], seed=cfg["data"]["shuffle_seed"]
    )
    ds_train, ds_val = dataset["train"], dataset["test"]
    print(f"{data_path}: {len(ds_train)} train, {len(ds_val)} val")

    denoiser = Denoiser(
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

    # The EMA is seeded with the prior, so it is live from the first step.
    tag = f"{cfg['model']['name']}_{cfg.get('params_tag', 'ema_latest')}"
    print(f"loading prior {tag} from {cfg['params_path']}")
    _, ema_params = load_model(cfg["params_path"], tag)
    ema_params = drop_cond_params(ema_params)
    if mesh is not None:
        ema_params = shard_state(ema_params, mesh)
    nnx.update(denoiser, ema_params)
    if mesh is not None:
        nnx.update(denoiser, shard_state(nnx.state(denoiser, nnx.Param), mesh))

    # The observation always comes in through `cond`: per sample in the loss, the
    # reference shear in log_corner.
    model = EPSDenoiser(denoiser, cfg, gamma=None, sigma_gamma=sigma_lsst)

    steps_per_epoch = len(ds_train) // cfg["training"]["batch_size"]
    total_steps = cfg["training"]["num_epochs"] * steps_per_epoch
    print(f"{total_steps} steps")

    optimizer = nnx.Optimizer(model, create_optimizer(cfg, total_steps), wrt=nnx.Param)
    loss_fn = EPSLoss(cfg, sigma_lsst)

    @nnx.jit
    def train_step(model, optimizer, x, cosmo, key):
        (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model=model, x=x, cosmo=cosmo, key=key, lambda_cosmo=cfg["loss"]["lambda_cosmo"], train=True
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
            loss = train_step(model, optimizer, x, cosmo, subkey)

            if use_ema:
                ema_params = update_ema(ema_params, nnx.state(denoiser, nnx.Param), cfg["ema"]["decay"])

            if step % cfg["logging"]["log_every_n_steps"] == 0:
                run.log({"train/loss_total": loss, "train/epoch": epoch})
            step += 1

        # Validate, visualize and checkpoint on the EMA weights. The live
        # parameters are copied out first, since nnx.state returns a view.
        if use_ema:
            live_params = jax.tree.map(lambda a: a.copy(), nnx.state(denoiser, nnx.Param))
            nnx.update(denoiser, ema_params)

        losses = []
        for i, batch in enumerate(val_loader):
            if val_max_batches is not None and i >= val_max_batches:
                break
            batch = normalize_batch(batch)
            key, val_key = jax.random.split(key, 2)
            losses.append(
                loss_fn(
                    model=model,
                    x=batch["map"],
                    cosmo=batch["theta"] * scale,
                    key=val_key,
                    lambda_cosmo=cfg["loss"]["lambda_cosmo"],
                    train=False,
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

        if corner is not None and (epoch + 1) % cfg["logging"]["visualize_every_n_epochs"] == 0:
            key, subkey = jax.random.split(key, 2)
            log_corner(cfg, model, subkey, corner)

        if use_ema:
            nnx.update(denoiser, live_params)

        if (epoch + 1) % cfg["checkpoint"]["save_every_n_epochs"] == 0:
            # Best is tracked on the cosmology loss, as in the other scripts.
            best = cfg["checkpoint"]["keep_best"] and val_loss_cosmo < best_cosmo_loss
            if best:
                best_cosmo_loss = val_loss_cosmo
            save_checkpoints(cfg, denoiser, ema_params if use_ema else None, checkpoint_dir, best)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the unconditional JADE prior into an EPS denoiser.")
    parser.add_argument("--config", type=str, required=True, help="e.g. configs/eps.yaml")
    train(load_config(parser.parse_args().config))
