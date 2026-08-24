#!/usr/bin/env python
"""Stage-2 training: refine cosmology pathways while freezing field reconstruction.

Loads a stage-1 checkpoint, splits each block's shared QKV into per-modality
copies (qkv_theta on cosmology tokens, qkv_kg on conditioning + field tokens),
then trains only the cosmology-side parameters per Strategy A (medium freeze):

  Trainable: cosmo_embedder, cosmo_head, blocks.*.attn.qkv_theta
  Frozen:    everything else (field/cond embedders, qkv_kg, mlp, ada_linear,
             norms, t_embedder, field_head, positional embeddings).

See experiments/configs/hybrid_stage2.yaml for the config schema.
"""

import os
import multiprocessing

if multiprocessing.current_process().name != "MainProcess":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import yaml
import pickle

import jax
import jax.numpy as jnp

from flax import nnx
import optax

import numpy as np

from datasets import load_from_disk
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from jade.nn_hybrid import JADE_B_16, convert_state_split_qkv
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, sigma_lsst
from jade.flow import Denoiser, FlowLoss
from jade.sampling import EulerSampler
from jade.utils import dump_model, load_model, plot_denoiser, plot_samples

from train_sharding import (
    augment,
    hf_collate,
    normalize_batch,
    make_cond,
    plot_corner,
)


# ============================================================================
# Freeze mask
# ============================================================================

# Substrings that mark a parameter as trainable in stage 2 (Strategy A: medium).
TRAINABLE_SUBSTRINGS_MEDIUM = (
    "cosmo_embedder",
    "cosmo_head",
    "qkv_theta",
)

# Strategy B (loose) would additionally include the V-projection on κγ tokens.
# Wiring it requires further splitting qkv_kg into q_kg/k_kg/v_kg — left for
# a follow-up. The substring 'v_kg' is a placeholder so the strategy code path
# exists.
TRAINABLE_SUBSTRINGS_LOOSE = TRAINABLE_SUBSTRINGS_MEDIUM + ("v_kg",)


def _path_to_str(path):
    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "name"):
            parts.append(str(p.name))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return ".".join(parts)


def make_freeze_labels(params, strategy="medium"):
    """Build a label pytree (matching ``params``) for ``optax.multi_transform``.

    Each leaf is labelled ``'train'`` if its path contains any of the strategy's
    trainable substrings, else ``'freeze'``.
    """
    if strategy == "medium":
        trainable = TRAINABLE_SUBSTRINGS_MEDIUM
    elif strategy == "loose":
        trainable = TRAINABLE_SUBSTRINGS_LOOSE
    else:
        raise ValueError(f"Unknown stage-2 strategy: {strategy}")

    def label(path, _leaf):
        ps = _path_to_str(path)
        return "train" if any(s in ps for s in trainable) else "freeze"

    return jax.tree_util.tree_map_with_path(label, params)


def summarize_freeze(params, labels):
    """Count trainable vs frozen elements and print a few examples of each."""
    leaves_p = jax.tree_util.tree_leaves_with_path(params)
    leaves_l = jax.tree_util.tree_leaves(labels)
    assert len(leaves_p) == len(leaves_l), "label/param tree mismatch"

    train_n, freeze_n = 0, 0
    train_ex, freeze_ex = [], []
    for (path, leaf), lbl in zip(leaves_p, leaves_l):
        size = leaf.value.size if hasattr(leaf, "value") else int(np.size(leaf))
        ps = _path_to_str(path)
        if lbl == "train":
            train_n += size
            if len(train_ex) < 6:
                train_ex.append((ps, size))
        else:
            freeze_n += size
            if len(freeze_ex) < 6:
                freeze_ex.append((ps, size))
    total = train_n + freeze_n
    print(f"\nFreeze summary ({total:,} total params):")
    print(f"  trainable: {train_n:,} ({100 * train_n / total:.2f}%)")
    print(f"  frozen:    {freeze_n:,} ({100 * freeze_n / total:.2f}%)")
    print("Trainable examples:")
    for p, s in train_ex:
        print(f"  + {p}  ({s:,})")
    print("Frozen examples:")
    for p, s in freeze_ex:
        print(f"  - {p}  ({s:,})")
    print()


# ============================================================================
# Optimizer
# ============================================================================


def create_stage2_optimizer(cfg, total_steps, labels):
    """AdamW with linear schedule, applied only to params labelled 'train'.

    Frozen params receive a zero update via ``optax.set_to_zero``. This means
    optimizer state for frozen params is allocated but never moves — slightly
    wasteful but unambiguous.
    """
    if cfg["optimizer"]["use_schedule"]:
        learning_rate = optax.linear_schedule(
            init_value=cfg["optimizer"]["schedule"]["init_value"],
            end_value=cfg["optimizer"]["schedule"]["end_value"],
            transition_steps=total_steps,
        )
    else:
        learning_rate = cfg["optimizer"]["learning_rate"]

    train_tx = optax.adamw(
        learning_rate=learning_rate,
        b1=cfg["optimizer"]["beta1"],
        b2=cfg["optimizer"]["beta2"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    return optax.chain(
        optax.clip_by_global_norm(cfg["optimizer"]["grad_clip_norm"]),
        optax.multi_transform(
            {"train": train_tx, "freeze": optax.set_to_zero()},
            labels,
        ),
    )


# ============================================================================
# Verification: frozen params must be byte-identical after one training step
# ============================================================================


def _flatten_for_compare(params):
    """Return a list of (path_str, jnp.ndarray) pairs for value-comparison."""
    out = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        arr = leaf.value if hasattr(leaf, "value") else leaf
        out.append((_path_to_str(path), np.asarray(arr)))
    return out


def verify_freeze(params_before, params_after, labels):
    """Assert that every leaf labelled 'freeze' is bit-identical pre/post-step."""
    before = _flatten_for_compare(params_before)
    after = _flatten_for_compare(params_after)
    flat_labels = jax.tree_util.tree_leaves(labels)
    assert len(before) == len(after) == len(flat_labels)

    n_frozen, n_drift = 0, 0
    n_train, n_train_moved = 0, 0
    drift_examples = []
    for (path_b, arr_b), (path_a, arr_a), lbl in zip(before, after, flat_labels):
        if lbl == "freeze":
            n_frozen += 1
            if not np.array_equal(arr_b, arr_a):
                n_drift += 1
                if len(drift_examples) < 5:
                    drift_examples.append((path_b, float(np.max(np.abs(arr_b - arr_a)))))
        else:
            n_train += 1
            if not np.array_equal(arr_b, arr_a):
                n_train_moved += 1

    print(f"\nFreeze verification:")
    print(f"  frozen leaves identical: {n_frozen - n_drift} / {n_frozen}")
    print(f"  trainable leaves moved:  {n_train_moved} / {n_train}")
    if n_drift > 0:
        print("  ✗ DRIFT in frozen params (max |Δ|):")
        for p, d in drift_examples:
            print(f"    {p}: {d:.3e}")
        raise AssertionError(f"{n_drift} frozen leaves drifted after one step")
    if n_train_moved == 0:
        raise AssertionError("No trainable leaf moved — optimizer not active?")
    print("  ✓ verification passed\n")


# ============================================================================
# Training
# ============================================================================


def train(cfg):
    # ------------------------------------------------------------------ setup
    print(f"\nDevices: {jax.device_count()} ({jax.devices()[0].platform})")

    if cfg["training"]["use_mixed_precision"]:
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        print(f"Mixed precision enabled: {cfg['training']['precision']}")

    run = wandb.init(
        project=cfg["logging"]["project"],
        entity=cfg["logging"]["entity"],
        config=cfg,
        tags=["stage2", cfg["stage2"]["strategy"]],
    )

    config_save_path = os.path.join(wandb.run.dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    checkpoint_dir = os.path.abspath(os.path.join(wandb.run.dir, "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ----------------------------------------------------------------- data
    num_workers = cfg["data"].get("num_workers", 8)
    dataset = load_from_disk(cfg["data"]["dataset_path"])
    keep_cols = [c for c in ("map", "theta") if c in dataset.column_names]
    dataset = dataset.select_columns(keep_cols).with_format("numpy")
    dataset = dataset.train_test_split(
        test_size=cfg["data"]["val_split"], seed=cfg["data"]["shuffle_seed"]
    )
    ds_train, ds_val = dataset["train"], dataset["test"]
    print(f"Train: {len(ds_train)}  Val: {len(ds_val)}  Workers: {num_workers}")

    # ---------------------------------------------------------------- model
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
        split_qkv=cfg["model"]["split_qkv"],
        mask_theta_to_field=cfg["model"].get("mask_theta_to_field", False),
    )
    model = Denoiser(model, cfg)

    # -------------------------------------------- load + convert stage-1 EMA
    params_dir = cfg["stage2"]["params_dir"]
    params_name = cfg["stage2"]["params_name"]
    print(f"Loading stage-1 checkpoint: {params_dir}/{params_name}")
    _, stage1_state = load_model(params_dir, params_name)
    stage2_state = convert_state_split_qkv(stage1_state)
    nnx.update(model, stage2_state)
    print("Stage-1 weights loaded into split-QKV model "
          "(qkv_theta and qkv_kg both set to stage-1 qkv).")

    # ``nnx.Optimizer.update`` internally strips Variable wrappers via
    # ``nnx.pure`` before calling optax (see flax/nnx/training/optimizer.py).
    # The label tree handed to ``optax.multi_transform`` must match that pure
    # structure, otherwise the mask tree carries ``Param`` nodes the updates
    # tree doesn't have and ``mask_pytree`` raises a custom-node-type mismatch.
    params = nnx.pure(nnx.state(model, nnx.Param))
    total = sum(x.size for x in jax.tree.leaves(params))
    print(f"Total parameters: {total:,}")

    # --------------------------------------------------- freeze mask & opt
    labels = make_freeze_labels(params, strategy=cfg["stage2"]["strategy"])
    summarize_freeze(params, labels)

    steps_per_epoch = len(ds_train) // cfg["training"]["batch_size"]
    total_steps = cfg["training"]["num_epochs"] * steps_per_epoch
    print(f"Total training steps: {total_steps}")

    opt = create_stage2_optimizer(cfg, total_steps, labels)
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    loss_fn = FlowLoss(cfg)

    # --------------------------------------------------- train step + EMA
    @nnx.jit
    def train_step(model, optimizer, x, cosmo, key, cond=None):
        (loss, (loss_x, loss_cosmo)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model=model, x=x, cosmo=cosmo, cond=cond, key=key,
            lambda_cosmo=cfg["loss"]["lambda_cosmo"], train=True,
        )
        optimizer.update(model, grads)
        return loss, loss_x, loss_cosmo

    @jax.jit
    def update_ema(ema_params, model_params, decay):
        return jax.tree.map(
            lambda ema, new: decay * ema + (1 - decay) * new, ema_params, model_params
        )

    # EMA restart from stage-1 EMA (= current model params, just loaded).
    if cfg["ema"]["use_ema"]:
        ema_params = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
        print(f"EMA restarted from stage-1 EMA (decay={cfg['ema']['decay']})")
    else:
        ema_params = None

    # ----------------------------------------------------- corner-plot ref
    mcmc_ref_dir = cfg.get("logging", {}).get("mcmc_ref_dir", "mcmc_log_normal")
    corner_obs, corner_mcmc, corner_truth = None, None, None
    try:
        with open(os.path.join(mcmc_ref_dir, "mcmc_log_obs_truth.pkl"), "rb") as f:
            ref = pickle.load(f)
        with open(os.path.join(mcmc_ref_dir, "mcmc_log_posterior_samples.pkl"), "rb") as f:
            corner_mcmc = pickle.load(f)
        corner_obs = jnp.asarray(ref["y"])
        corner_truth = np.asarray(ref["theta"])
        print(f"Corner-plot reference loaded from {mcmc_ref_dir}")
    except FileNotFoundError as e:
        print(f"Corner-plot reference not found ({e}); skipping corner plots")

    # ----------------------------------------------------- data loaders
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=hf_collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        pin_memory=False,
        multiprocessing_context="forkserver" if num_workers > 0 else None,
    )

    # ---------------------------------------------- verification step
    if cfg["stage2"].get("verify_freeze", True):
        print("Running freeze-verification step...")
        params_before = jax.tree.map(lambda x: np.asarray(x), nnx.state(model, nnx.Param))

        # Pull a single batch and take exactly one training step
        verify_batch = next(iter(train_loader))
        verify_batch = normalize_batch(verify_batch)
        v_key = jax.random.key(0)
        v_key, sk1 = jax.random.split(v_key)
        x_v = augment(verify_batch["map"], jax.random.split(sk1, len(verify_batch["map"])))
        cosmo_v = verify_batch["theta"] * cfg["loss"]["SCALE_COSMO"]
        v_key, sk2 = jax.random.split(v_key)
        cond_v = make_cond(x_v, sk2) if cfg["model"]["enable_cond_image"] else None
        v_key, sk3 = jax.random.split(v_key)
        _ = train_step(model, optimizer, x_v, cosmo_v, sk3, cond=cond_v)

        params_after = jax.tree.map(lambda x: np.asarray(x), nnx.state(model, nnx.Param))
        verify_freeze(params_before, params_after, labels)

        # Restore: reload stage-1 weights so the actual training run starts from
        # the converted checkpoint, not the post-verification-step state.
        nnx.update(model, stage2_state)
        if cfg["ema"]["use_ema"]:
            ema_params = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
        # Optimizer state survives the restore — we re-initialise it here so
        # AdamW moments don't carry over from the verification step.
        optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)
        print("Restored stage-1 weights and re-initialised optimizer.\n")

    # ----------------------------------------------------- training loop
    key = jax.random.key(cfg["training"]["seed"])
    best_val_loss_cosmo = float("inf")
    step = 0
    num_epochs = cfg["training"]["num_epochs"]

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch = normalize_batch(batch)

            key, sk = jax.random.split(key)
            x = augment(batch["map"], jax.random.split(sk, len(batch["map"])))
            cosmo = batch["theta"] * cfg["loss"]["SCALE_COSMO"]

            key, sk = jax.random.split(key)
            cond_field = make_cond(x, sk) if cfg["model"]["enable_cond_image"] else None

            key, sk = jax.random.split(key)
            loss, loss_x, loss_cosmo = train_step(model, optimizer, x, cosmo, sk, cond=cond_field)

            if cfg["ema"]["use_ema"] and ema_params is not None:
                ema_params = update_ema(
                    ema_params, nnx.state(model, nnx.Param), cfg["ema"]["decay"]
                )

            if step % cfg["logging"]["log_every_n_steps"] == 0:
                run.log({
                    "train/loss_total": float(loss),
                    "train/loss_x": float(loss_x),
                    "train/loss_cosmo": float(loss_cosmo),
                    "train/epoch": epoch,
                })
            step += 1

        # ----------------------------------------------------- validation
        if cfg["ema"]["use_ema"] and ema_params is not None:
            original_params = nnx.state(model, nnx.Param)
            nnx.update(model, ema_params)

        losses, losses_x, losses_cosmo = [], [], []
        val_loader = DataLoader(
            ds_val,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=hf_collate,
            drop_last=False,
            persistent_workers=num_workers > 0,
            pin_memory=False,
            multiprocessing_context="forkserver" if num_workers > 0 else None,
        )
        for vb in val_loader:
            vb = normalize_batch(vb)
            x_val = vb["map"]
            cosmo_val = vb["theta"] * cfg["loss"]["SCALE_COSMO"]
            key, vk = jax.random.split(key)
            cond_val = make_cond(x_val, vk) if cfg["model"]["enable_cond_image"] else None
            v_loss, (v_lx, v_lc) = loss_fn(
                model=model, x=x_val, cosmo=cosmo_val, key=vk,
                lambda_cosmo=cfg["loss"]["lambda_cosmo"], train=False, cond=cond_val,
            )
            losses.append(v_loss)
            losses_x.append(v_lx)
            losses_cosmo.append(v_lc)

        val_loss = float(np.mean(losses))
        val_loss_x = float(np.mean(losses_x))
        val_loss_cosmo = float(np.mean(losses_cosmo))
        run.log({
            "val/loss_total": val_loss,
            "val/v_loss_field": val_loss_x,
            "val/v_loss_cosmo": val_loss_cosmo,
            "epoch": epoch + 1,
        })
        print(f"Epoch {epoch + 1}: val loss {val_loss:.4f}  "
              f"(v_field {val_loss_x:.4f}, v_cosmo {val_loss_cosmo:.4f})")

        # ----------------------------------------------------- viz
        if (epoch + 1) % cfg["logging"]["visualize_every_n_epochs"] == 0:
            fig = plot_denoiser(x_val, cosmo_val, model, key, cfg)
            wandb.log({"denoiser_ema": wandb.Image(fig)})
            plt.close(fig)

            key, sk = jax.random.split(key)
            sampler = EulerSampler(model=model, num_steps=50)
            keys = jax.random.split(sk, 3)
            x_0 = jax.random.normal(keys[0], shape=(6, 128, 128, 5))
            cosmo_0 = jax.random.normal(keys[1], shape=(6, 6))
            sk_keys = jax.random.split(keys[2], 6)
            cond_plot = (
                make_cond(x_val[:6], keys[2]) if cfg["model"]["enable_cond_image"] else None
            )
            x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, cond_plot, sk_keys)
            fig = plot_samples(x_samples, cosmo_samples / cfg["loss"]["SCALE_COSMO"], n_samples=6)
            wandb.log({"samples": wandb.Image(fig)})
            plt.close(fig)

            if corner_obs is not None and cfg["model"]["enable_cond_image"]:
                key, sk = jax.random.split(key)
                n_corner = 128
                corner_keys = jax.random.split(sk, 3)
                x0_c = jax.random.normal(corner_keys[0], shape=(n_corner, 128, 128, 5))
                cosmo0_c = jax.random.normal(corner_keys[1], shape=(n_corner, 6))
                vmap_keys = jax.random.split(corner_keys[2], n_corner)
                cond_c = (corner_obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)
                _, cosmo_post = jax.vmap(sampler, in_axes=(0, 0, None, 0))(
                    x0_c, cosmo0_c, cond_c, vmap_keys
                )
                theta_post = (
                    np.asarray(cosmo_post) / cfg["loss"]["SCALE_COSMO"] * THETA_STD + THETA_MEAN
                )
                fig = plot_corner(theta_post, corner_mcmc, corner_truth)
                wandb.log({"corner": wandb.Image(fig)})
                plt.close("all")

        if cfg["ema"]["use_ema"] and ema_params is not None:
            nnx.update(model, original_params)

        # ----------------------------------------------------- checkpoint
        if (epoch + 1) % cfg["checkpoint"]["save_every_n_epochs"] == 0:
            if cfg["ema"]["use_ema"] and ema_params is not None:
                dump_model(cfg, ema_params, f"{cfg['model']['name']}_ema_latest", checkpoint_dir)
            dump_model(cfg, nnx.state(model, nnx.Param),
                       f"{cfg['model']['name']}_latest", checkpoint_dir)

            if cfg["checkpoint"]["keep_best"] and val_loss_cosmo < best_val_loss_cosmo:
                best_val_loss_cosmo = val_loss_cosmo
                if cfg["ema"]["use_ema"] and ema_params is not None:
                    dump_model(cfg, ema_params, f"{cfg['model']['name']}_ema_best", checkpoint_dir)
                dump_model(cfg, nnx.state(model, nnx.Param),
                           f"{cfg['model']['name']}_best", checkpoint_dir)
                print(f"  Best model saved (val_loss_cosmo: {val_loss_cosmo:.4f})")
            print("  Checkpoints saved")

    print("Stage 2 training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-2 fine-tuning for JADE.")
    parser.add_argument("--config", type=str, default="configs/hybrid_stage2.yaml",
                        help="Path to stage-2 config")
    parser.add_argument("--params-dir", type=str, default=None,
                        help="Override stage2.params_dir from config")
    parser.add_argument("--params-name", type=str, default=None,
                        help="Override stage2.params_name from config")
    args = parser.parse_args()

    def parse_config(c):
        if isinstance(c, dict):
            return {k: parse_config(v) for k, v in c.items()}
        if isinstance(c, list):
            return [parse_config(v) for v in c]
        if isinstance(c, str):
            try:
                if "." in c or "e" in c.lower():
                    return float(c)
                return int(c)
            except ValueError:
                return c
        return c

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = parse_config(cfg)

    if args.params_dir is not None:
        cfg["stage2"]["params_dir"] = args.params_dir
    if args.params_name is not None:
        cfg["stage2"]["params_name"] = args.params_name

    print("=" * 60)
    print("Stage 2 configuration:")
    print("=" * 60)
    import pprint
    pprint.pprint(cfg)
    print("=" * 60)

    train(cfg)
