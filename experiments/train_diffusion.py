#!/usr/bin/env python

import os
import multiprocessing
# DataLoader workers (spawn/forkserver) re-import this module; keep them off GPU
# so they don't each try to grab a CUDA context and OOM.
if multiprocessing.current_process().name != "MainProcess":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import yaml
import gc
import pickle

import jax
import jax.numpy as jnp

from flax import nnx
import optax

import dm_pix as pix

import numpy as np

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

from datasets import load_from_disk
from functools import partial

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from getdist import MCSamples, plots

from jade.nn_hybrid import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, sigma_lsst
from jade.diffusion import Denoiser, DenoiserLoss, HeunSampler
from jade.utils import dump_model, load_model, plot_samples


@jax.jit
@jax.vmap
def augment(x, key):
    """x: image [w, h, c]; key: random key."""
    keys = jax.random.split(key, 2)
    x = pix.random_flip_left_right(keys[0], x)
    x = pix.random_flip_up_down(keys[1], x)
    return x


def hf_collate(batch):
    return {
        "map": np.stack([b["map"] for b in batch]),
        "theta": np.stack([b["theta"] for b in batch]),
    }


@jax.jit
def normalize_batch(batch):
    theta_norm = (batch['theta'] - THETA_MEAN) / THETA_STD
    field_mean = FIELD_MEAN.reshape(1, 1, 1, -1)
    field_std = FIELD_STD.reshape(1, 1, 1, -1)
    map_norm = (batch['map'] - field_mean) / field_std
    return {'map': map_norm, 'theta': theta_norm}


@jax.jit
def make_cond(x, key):
    x = x * FIELD_STD.reshape(1, 1, 1, -1) + FIELD_MEAN.reshape(1, 1, 1, -1)
    cond = x + sigma_lsst.reshape((1, 1, 1, -1)) * jax.random.normal(key, shape=x.shape)
    cond = (cond - FIELD_MEAN.reshape(1, 1, 1, -1)) / FIELD_STD.reshape(1, 1, 1, -1)
    return cond


def plot_corner(theta_post, mcmc_samples, theta_truth):
    """Triangle plot comparing diffusion posterior to reference MCMC samples."""
    names = [r"$\Omega_c$", r"$\Omega_b$", r"$\sigma_8$", r"$h_0$", r"$n_s$", r"$w_0$"]
    s_post = MCSamples(samples=np.asarray(theta_post), names=names, label="Diffusion")
    s_mcmc = MCSamples(samples=np.asarray(mcmc_samples), names=names, label="MCMC")

    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 14
    g.settings.axes_labelsize = 16
    g.settings.legend_fontsize = 16
    g.triangle_plot(
        [s_post, s_mcmc],
        names,
        markers=np.asarray(theta_truth),
        marker_args={"lw": 1},
        filled=[True, False],
        contour_colors=["#d06e99ff", "black"],
        contour_ls=["-", "--"],
        contour_lws=[2., 2.],
    )
    return plt.gcf()


def plot_denoiser_edm(x, cosmo, model, key, cfg):
    """Visualization for an EDM denoiser at one sampled sigma per example."""
    keys = jax.random.split(key, 4)

    # Draw a single sigma per example from the same log-normal used in training
    P_mean = cfg["diffusion"].get("P_mean", -1.2)
    P_std = cfg["diffusion"].get("P_std", 1.2)
    log_sigma = P_mean + P_std * jax.random.normal(keys[0], shape=x.shape[:1])
    sigma = jnp.exp(log_sigma)

    sigma_b_x = sigma[:, None, None, None]
    sigma_b_c = sigma[:, None]

    noise_x = jax.random.normal(keys[1], shape=x.shape)
    noise_c = jax.random.normal(keys[2], shape=cosmo.shape)
    xt = x + sigma_b_x * noise_x
    cosmot = cosmo + sigma_b_c * noise_c

    cond = make_cond(x, keys[3]) if cfg["model"]["enable_cond_image"] else None

    dummy_keys = jnp.zeros((x.shape[0], 2), dtype=jnp.uint32)
    x_pred, cosmo_pred = jax.vmap(
        model.x_pred, in_axes=(0, 0, 0, 0, None, 0)
    )(xt, cosmot, sigma, cond, False, dummy_keys)

    def denormalize(cosmo_norm):
        return cosmo_norm * THETA_STD + THETA_MEAN

    cosmo_denorm = denormalize(cosmo)
    cosmot_denorm = denormalize(cosmot)
    cosmo_pred_denorm = denormalize(cosmo_pred)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.5],
                          hspace=0.3, wspace=0.2)

    for i in range(5):
        for j in range(3):
            ax = fig.add_subplot(gs[j, i])
            if j == 0:
                ax.imshow(x[0, ..., i], cmap='viridis')
                if i == 0:
                    ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
            elif j == 1:
                ax.imshow(xt[0, ..., i], cmap='viridis')
                if i == 0:
                    ax.set_ylabel(f'Noisy (σ={float(sigma[0]):.2f})',
                                  fontsize=12, fontweight='bold')
            else:
                ax.imshow(x_pred[0, ..., i], cmap='viridis')
                if i == 0:
                    ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
            ax.axis('off')

    param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w0']
    for j in range(3):
        ax_text = fig.add_subplot(gs[j, 5])
        ax_text.axis('off')
        if j == 0:
            cosmo_vals = cosmo_denorm[0]; title = 'Ground Truth\nCosmology'
        elif j == 1:
            cosmo_vals = cosmot_denorm[0]; title = 'Noisy\nCosmology'
        else:
            cosmo_vals = cosmo_pred_denorm[0]; title = 'Predicted\nCosmology'

        text = f'{title}\n' + '─' * 15 + '\n'
        for name, val in zip(param_names, cosmo_vals):
            text += f'{name:>4s}: {val:7.4f}\n'
        ax_text.text(0.1, 0.5, text, fontsize=10, family='monospace',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    return fig


def create_optimizer(cfg, total_steps):
    if cfg['optimizer']['use_schedule']:
        schedule = optax.linear_schedule(
            init_value=cfg['optimizer']['schedule']['init_value'],
            end_value=cfg['optimizer']['schedule']['end_value'],
            transition_steps=total_steps,
        )
        learning_rate = schedule
    else:
        learning_rate = cfg['optimizer']['learning_rate']

    opt = optax.chain(
        optax.clip_by_global_norm(cfg['optimizer']['grad_clip_norm']),
        optax.adamw(
            learning_rate=learning_rate,
            b1=cfg['optimizer']['beta1'],
            b2=cfg['optimizer']['beta2'],
            weight_decay=cfg['optimizer']['weight_decay']
        ),
    )
    return opt


def setup_mesh_and_sharding(num_devices):
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=('fsdp',))
    print(f"Created mesh with {num_devices} devices: {mesh}")
    print(f"Device IDs: {mesh.devices}")
    sharding_strategy = {
        'params': P('fsdp'),
        'batch': P(None),
    }
    return mesh, sharding_strategy


def shard_model_state(state, mesh, spec):
    num_devices = mesh.shape['fsdp']

    def create_sharding(x):
        if x.ndim == 0 or x.size == 0:
            return NamedSharding(mesh, P())
        if x.ndim == 1:
            if x.shape[0] % num_devices == 0:
                return NamedSharding(mesh, spec)
            else:
                return NamedSharding(mesh, P())
        if x.ndim == 2:
            if x.shape[0] % num_devices == 0:
                return NamedSharding(mesh, spec)
            elif x.shape[1] % num_devices == 0:
                return NamedSharding(mesh, P(None, 'fsdp'))
            else:
                return NamedSharding(mesh, P())
        if x.ndim == 4:
            if x.shape[-1] % num_devices == 0:
                return NamedSharding(mesh, P(None, None, None, 'fsdp'))
            elif x.shape[-2] % num_devices == 0:
                return NamedSharding(mesh, P(None, None, 'fsdp', None))
            else:
                return NamedSharding(mesh, P())
        for i, dim in enumerate(x.shape):
            if dim % num_devices == 0:
                sharding_spec = [None] * x.ndim
                sharding_spec[i] = 'fsdp'
                return NamedSharding(mesh, P(*sharding_spec))
        return NamedSharding(mesh, P())

    shardings = jax.tree.map(create_sharding, state)
    sharded_state = jax.tree.map(
        lambda x, s: jax.device_put(x, s),
        state,
        shardings,
    )
    return sharded_state


def train(cfg):
    """Main EDM diffusion training function with FSDP support."""

    num_devices = jax.device_count()
    print(f"\n{'='*60}")
    print(f"Multi-GPU Training Setup (EDM diffusion)")
    print(f"{'='*60}")
    print(f"Available devices: {num_devices}")
    print(f"Device type: {jax.devices()[0].platform}")

    use_sharding = cfg.get('distributed', {}).get('use_sharding', False)
    if use_sharding and num_devices > 1:
        print(f"✓ FSDP sharding ENABLED across {num_devices} devices")
        mesh, sharding_strategy = setup_mesh_and_sharding(num_devices)
    else:
        if num_devices > 1:
            print(f"⚠ Multiple devices detected but sharding DISABLED")
            print(f"  Set 'distributed.use_sharding: true' in config to enable")
        mesh = None
        sharding_strategy = None
    print(f"{'='*60}\n")

    if cfg['training']['use_mixed_precision']:
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
        print(f"Mixed precision enabled: {cfg['training']['precision']}")

    # EDM-specific config print
    print(f"EDM diffusion:")
    print(f"  P_mean: {cfg['diffusion'].get('P_mean', -1.2)}")
    print(f"  P_std:  {cfg['diffusion'].get('P_std', 1.2)}")
    print(f"  sigma_min/max: {cfg['diffusion'].get('sigma_min', 2e-3)} / "
          f"{cfg['diffusion'].get('sigma_max', 80.0)}")
    print(f"  sigma_data (x/cosmo): {cfg['diffusion'].get('sigma_data_x', 1.0)} / "
          f"{cfg['diffusion'].get('sigma_data_cosmo', 1.0)}")

    run = wandb.init(
        project=cfg['logging']['project'],
        entity=cfg['logging']['entity'],
        config=cfg,
    )

    config_save_path = os.path.join(wandb.run.dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    checkpoint_dir = os.path.abspath(os.path.join(wandb.run.dir, 'checkpoints'))
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_workers = cfg['data'].get('num_workers', 8)
    dataset = load_from_disk(cfg['data']['dataset_path'])
    keep_cols = [c for c in ("map", "theta") if c in dataset.column_names]
    dataset = dataset.select_columns(keep_cols)
    dataset = dataset.with_format("numpy")

    dataset = dataset.train_test_split(
        test_size=cfg['data']['val_split'],
        seed=cfg['data']['shuffle_seed']
    )
    ds_train = dataset["train"]
    ds_val = dataset["test"]
    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples: {len(ds_val)}")
    print(f"DataLoader workers: {num_workers}")

    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    model = JADE_B_16(
        rngs=nnx.Rngs(cfg['training']['seed']),
        in_channels=cfg['model']['in_channels'],
        input_size=cfg['model']['input_size'],
        enable_cond_image=cfg["model"]["enable_cond_image"],
        cond_channels=cfg["model"]["cond_channels"],
        num_cosmo_tokens=cfg['model']['num_cosmo_tokens'],
        cond_patch_size=cfg['model']['cond_patch_size'],
        cond_start=cfg['model']['cond_start'],
        attn_drop=cfg['model']['attn_drop'],
        proj_drop=cfg['model']['proj_drop'],
    )

    # Wrap with EDM-preconditioned denoiser
    model = Denoiser(model, cfg)

    # ========================================================================
    # CHECKPOINT LOADING
    # ========================================================================
    if cfg['start_from_checkpoint']:
        params_tag = cfg.get('params_tag', 'ema_latest')
        print(f"Loading model from checkpoint: {cfg['params_path']} (tag: {params_tag})")
        _, ema_params = load_model(cfg['params_path'], f"{cfg['model']['name']}_{params_tag}")
        if use_sharding and mesh is not None:
            print("Re-sharding loaded checkpoint to match current mesh...")
            ema_params = shard_model_state(
                ema_params,
                mesh,
                sharding_strategy['params']
            )
        nnx.update(model, ema_params)
    else:
        if cfg['ema']['use_ema']:
            ema_params = None
            print(f"EMA will be initialized after first epoch")
        else:
            ema_params = None
            print("EMA disabled")

    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree.leaves(params))
    print(f"Total parameters: {total:,}")

    steps_per_epoch = len(ds_train) // cfg['training']['batch_size']
    total_steps = cfg['training']['num_epochs'] * steps_per_epoch
    print(f"Total training steps: {total_steps}")

    # ========================================================================
    # SHARD PARAMETERS BEFORE OPTIMIZER
    # ========================================================================
    if use_sharding and mesh is not None:
        print("\nSharding model parameters...")
        model_params = nnx.state(model, nnx.Param)
        model_params_sharded = shard_model_state(
            model_params,
            mesh,
            sharding_strategy['params']
        )
        nnx.update(model, model_params_sharded)
        print("✓ Model parameters sharded across devices")

        print("\n" + "="*60)
        print("Verifying Parameter Sharding:")
        print("="*60)

        params_list = list(jax.tree_util.tree_leaves_with_path(model_params_sharded))

        sharded_count = 0
        replicated_count = 0
        sharded_params = 0
        replicated_params = 0
        for key, param in params_list:
            spec = param.sharding.spec if hasattr(param.sharding, 'spec') else None
            if spec is not None and spec != P():
                sharded_count += 1
                sharded_params += param.size
            else:
                replicated_count += 1
                replicated_params += param.size

        print(f"Sharded parameters: {sharded_count} arrays ({sharded_params:,} elements)")
        print(f"Replicated parameters: {replicated_count} arrays ({replicated_params:,} elements)")
        total_p = sharded_params + replicated_params
        print(f"Total sharding: {sharded_params / total_p * 100:.1f}% of parameters\n")

        def get_path_string(key):
            parts = []
            for k in key:
                if hasattr(k, 'key'):
                    parts.append(str(k.key))
                else:
                    parts.append(str(k).split('(')[0] if '(' in str(k) else str(k))
            return '.'.join(parts)

        print("Sample sharded parameters:")
        shown = 0
        for key, param in params_list:
            if shown >= 3:
                break
            spec = param.sharding.spec if hasattr(param.sharding, 'spec') else None
            if spec is not None and spec != P():
                path = get_path_string(key)
                print(f"  {path}:")
                print(f"    Shape: {param.shape}")
                print(f"    Sharding: {param.sharding.spec}")
                shown += 1

        print("\nSample replicated parameters:")
        shown = 0
        for key, param in params_list:
            if shown >= 3:
                break
            spec = param.sharding.spec if hasattr(param.sharding, 'spec') else None
            if spec is None or spec == P():
                path = get_path_string(key)
                print(f"  {path}:")
                print(f"    Shape: {param.shape}")
                print(f"    Sharding: replicated")
                shown += 1
        print("="*60 + "\n")

    # ========================================================================
    # OPTIMIZER
    # ========================================================================
    opt = create_optimizer(cfg, total_steps)
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    if use_sharding and mesh is not None:
        print("✓ Optimizer state automatically sharded to match parameters\n")

    loss_fn = DenoiserLoss(cfg)

    # ========================================================================
    # TRAINING STEP
    # ========================================================================
    @nnx.jit
    def train_step(model, optimizer, x, cosmo, key, cond=None):
        (loss, (_, _)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model=model, x=x, cosmo=cosmo, cond=cond, key=key, train=True
        )
        optimizer.update(model, grads)
        return loss

    @jax.jit
    def update_ema(ema_params, model_params, decay):
        return jax.tree.map(
            lambda ema, new: decay * ema + (1 - decay) * new,
            ema_params,
            model_params
        )

    key = jax.random.key(cfg['training']['seed'])
    best_val_loss = float('inf')
    step = 0

    lap = 0 if not cfg['start_from_checkpoint'] else cfg['lap']
    num_epochs = cfg['training']['num_epochs']

    # ========================================================================
    # CORNER-PLOT REFERENCE
    # ========================================================================
    mcmc_ref_dir = cfg.get('logging', {}).get('mcmc_ref_dir', 'mcmc_log_normal')
    corner_obs = None
    corner_mcmc = None
    corner_truth = None
    try:
        with open(os.path.join(mcmc_ref_dir, 'mcmc_log_obs_truth.pkl'), 'rb') as f:
            ref = pickle.load(f)
        with open(os.path.join(mcmc_ref_dir, 'mcmc_log_posterior_samples.pkl'), 'rb') as f:
            corner_mcmc = pickle.load(f)
        corner_obs = jnp.asarray(ref['y'])
        corner_truth = np.asarray(ref['theta'])
        print(f"Corner-plot reference loaded from {mcmc_ref_dir}")
    except FileNotFoundError as e:
        print(f"Corner-plot reference not found ({e}); skipping corner plots")

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=hf_collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        pin_memory=False,
        multiprocessing_context='forkserver' if num_workers > 0 else None,
    )

    sampler_num_steps = cfg.get('sampling', {}).get('num_steps', 18)
    sigma_min_s = cfg['diffusion'].get('sigma_min', 2e-3)
    sigma_max_s = cfg['diffusion'].get('sigma_max', 80.0)
    rho_s = cfg.get('sampling', {}).get('rho', 7.0)

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['num_epochs']}"):

            batch = normalize_batch(batch)

            key, subkey = jax.random.split(key, 2)
            x = augment(batch["map"], jax.random.split(subkey, len(batch["map"])))
            cosmo = batch["theta"] * cfg['loss']['SCALE_COSMO']

            key, subkey = jax.random.split(key, 2)
            if cfg["model"]["enable_cond_image"]:
                cond_field = make_cond(x, subkey)
            else:
                cond_field = None

            key, subkey = jax.random.split(key, 2)
            loss = train_step(model, optimizer, x, cosmo, subkey, cond=cond_field)

            # ----------------------------------------------------------------
            # EMA INITIALIZATION (with sharding)
            # ----------------------------------------------------------------
            if cfg['ema']['use_ema'] and ema_params is None and epoch >= 1 and not cfg["start_from_checkpoint"]:
                ema_params = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
                if use_sharding and mesh is not None:
                    ema_params = shard_model_state(
                        ema_params,
                        mesh,
                        sharding_strategy['params']
                    )
                print(f"EMA initialized from trained model at epoch {epoch}")

            if cfg['ema']['use_ema'] and ema_params is not None:
                current_params = nnx.state(model, nnx.Param)
                ema_params = update_ema(ema_params, current_params, cfg['ema']['decay'])

            if step % cfg['logging']['log_every_n_steps'] == 0:
                run.log({"train/loss_total": loss, "train/epoch": epoch})

            step += 1

        # ====================================================================
        # VALIDATION
        # ====================================================================
        if cfg['ema']['use_ema'] and ema_params is not None:
            original_params = nnx.state(model, nnx.Param)
            nnx.update(model, ema_params)

        losses = []
        losses_x = []
        losses_cosmo = []
        val_loader = DataLoader(
            ds_val,
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=hf_collate,
            drop_last=False,
            persistent_workers=num_workers > 0,
            pin_memory=False,
            multiprocessing_context='forkserver' if num_workers > 0 else None,
        )
        for batch in val_loader:
            batch = normalize_batch(batch)
            x_val = batch["map"]
            cosmo_val = batch["theta"] * cfg['loss']['SCALE_COSMO']

            key, val_key = jax.random.split(key, 2)

            if cfg["model"]["enable_cond_image"]:
                cond_val = make_cond(x_val, val_key)
            else:
                cond_val = None

            val_loss, (val_loss_x, val_loss_cosmo) = loss_fn(
                model=model, x=x_val, cosmo=cosmo_val, key=val_key,
                train=False, cond=cond_val,
            )
            losses.append(val_loss)
            losses_x.append(val_loss_x)
            losses_cosmo.append(val_loss_cosmo)

        val_loss = np.mean(losses)
        val_loss_x = np.mean(losses_x)
        val_loss_cosmo = np.mean(losses_cosmo)

        run.log({
            "val/loss_total": val_loss,
            "val/loss_field": val_loss_x,
            "val/loss_cosmo": val_loss_cosmo,
            "epoch": epoch + 1,
        })

        model_type = "EMA" if cfg['ema']['use_ema'] else "Standard"
        print(f"Epoch {epoch + 1}, Val Loss ({model_type}): {val_loss:.4f} "
              f"(field: {val_loss_x:.4f}, cosmo: {val_loss_cosmo:.4f})")

        # Visualization
        if (epoch + 1) % cfg['logging']['visualize_every_n_epochs'] == 0:
            fig = plot_denoiser_edm(x_val, cosmo_val, model, key, cfg)
            tag = "denoiser_ema" if cfg['ema']['use_ema'] else "denoiser"
            wandb.log({tag: wandb.Image(fig)})
            plt.close(fig)

            key, subkey = jax.random.split(key, 2)

            sampler = HeunSampler(
                model=model,
                num_steps=sampler_num_steps,
                sigma_min=sigma_min_s,
                sigma_max=sigma_max_s,
                rho=rho_s,
            )

            keys = jax.random.split(subkey, 3)
            x_0 = jax.random.normal(keys[0], shape=(6, 128, 128, 5))
            cosmo_0 = jax.random.normal(keys[1], shape=(6, 6))

            keys = jax.random.split(keys[2], 6)
            if cfg["model"]["enable_cond_image"]:
                cond_plot = make_cond(x_val[:6], keys[2])
            else:
                cond_plot = None

            x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, cond_plot, keys)

            fig = plot_samples(x_samples, cosmo_samples / cfg['loss']['SCALE_COSMO'], n_samples=6)
            wandb.log({"samples": wandb.Image(fig)})
            plt.close(fig)

            if corner_obs is not None and cfg["model"]["enable_cond_image"]:
                key, subkey = jax.random.split(key, 2)
                n_corner = 128
                corner_keys = jax.random.split(subkey, 3)
                x0_c = jax.random.normal(corner_keys[0], shape=(n_corner, 128, 128, 5))
                cosmo0_c = jax.random.normal(corner_keys[1], shape=(n_corner, 6))
                vmap_keys = jax.random.split(corner_keys[2], n_corner)
                cond_c = (corner_obs - FIELD_MEAN.reshape(1, 1, -1)) / FIELD_STD.reshape(1, 1, -1)
                _, cosmo_post = jax.vmap(sampler, in_axes=(0, 0, None, 0))(
                    x0_c, cosmo0_c, cond_c, vmap_keys
                )
                theta_post = np.asarray(cosmo_post) / cfg['loss']['SCALE_COSMO'] * THETA_STD + THETA_MEAN
                fig = plot_corner(theta_post, corner_mcmc, corner_truth)
                wandb.log({"corner": wandb.Image(fig)})
                plt.close('all')

        if cfg['ema']['use_ema'] and ema_params is not None:
            nnx.update(model, original_params)

        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        if (epoch + 1) % cfg['checkpoint']['save_every_n_epochs'] == 0:
            if cfg['ema']['use_ema'] and ema_params is not None:
                dump_model(
                    cfg,
                    ema_params,
                    f"{cfg['model']['name']}_ema_latest",
                    checkpoint_dir,
                )
            dump_model(
                cfg,
                nnx.state(model, nnx.Param),
                f"{cfg['model']['name']}_latest",
                checkpoint_dir,
            )

            if cfg['checkpoint']['keep_best'] and val_loss_cosmo < best_val_loss:
                best_val_loss = val_loss_cosmo
                if cfg['ema']['use_ema'] and ema_params is not None:
                    dump_model(
                        cfg,
                        ema_params,
                        f"{cfg['model']['name']}_ema_best",
                        checkpoint_dir,
                    )
                dump_model(
                    cfg,
                    nnx.state(model, nnx.Param),
                    f"{cfg['model']['name']}_best",
                    checkpoint_dir,
                )
                print(f"✓ Best model saved (val_loss: {val_loss:.4f})")

            print(f"✓ Checkpoints saved")

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the JADE EDM-diffusion model with optional FSDP.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    def parse_config(cfg):
        if isinstance(cfg, dict):
            return {k: parse_config(v) for k, v in cfg.items()}
        elif isinstance(cfg, list):
            return [parse_config(v) for v in cfg]
        elif isinstance(cfg, str):
            try:
                if '.' in cfg or 'e' in cfg.lower():
                    return float(cfg)
                else:
                    return int(cfg)
            except ValueError:
                return cfg
        else:
            return cfg

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = parse_config(cfg)

    print("="*50)
    print("Configuration:")
    print("="*50)
    import pprint
    pprint.pprint(cfg)
    print("="*50)

    train(cfg)
