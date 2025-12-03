#!/usr/bin/env python

import os
import argparse
import yaml

import jax
import jax.numpy as jnp

from flax import nnx
import orbax.checkpoint as ocp
import optax

import dm_pix as pix

import numpy as np

from datasets import load_from_disk
from functools import partial

import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from jade.nn import JADE_B_16, JADE_L_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD  # Import normalization stats
from jade.diffusion import Denoiser

def sigma_fn(t, cfg):
    """Noise schedule function"""
    return cfg['diffusion']['sigma_min'] * (
        cfg['diffusion']['sigma_max'] / cfg['diffusion']['sigma_min']
    ) ** t


def get_weight_fn(cfg):
    """Get weight function based on config"""
    if cfg['diffusion']['weight_type'] == "inverse_sigma_squared_plus_one":
        return lambda t: 1 / sigma_fn(t, cfg)**2 + 1
    elif cfg['diffusion']['weight_type'] == "uniform":
        return lambda t: 1.0
    else:
        raise ValueError(f"Unknown weight type: {cfg['diffusion']['weight_type']}")

@jax.jit
@jax.vmap
def augment(x, key):
    """
    x: image [w, h, c]
    key: random key
    """
    keys = jax.random.split(key, 2)
    x = pix.random_flip_left_right(keys[0], x)
    x = pix.random_flip_up_down(keys[1], x)
    return x

def plot_denoiser(x, cosmo, model, key, cfg):
    """Visualization function"""
    keys = jax.random.split(key, 3)
    
    if cfg['diffusion']['time_distribution'] == "logit":
        mu = cfg['diffusion']['mu']
        sigma = cfg['diffusion']['sigma']
        s = (jax.random.normal(keys[0], shape=x.shape[:1]) + mu) * sigma
        t = jax.nn.sigmoid(s)

    elif cfg['diffusion']['time_distribution'] == "beta":
        t = jax.random.beta(
            keys[0], 
            a=cfg['diffusion']['beta_a'], 
            b=cfg['diffusion']['beta_b'], 
            shape=x.shape[:1]
        )
    else:
        t = jax.random.uniform(keys[0], shape=x.shape[:1])
    
    # sigma_t = sigma_fn(t, cfg)
    
    # forward diffusion
    # z = sigma_t[...,None,None,None] * jax.random.normal(keys[1], shape=x.shape)
    # xt = x + z

    # zc = sigma_t[...,None] * jax.random.normal(keys[2], shape=cosmo.shape)
    # cosmot = cosmo + zc
    
    t = t*0. + 0.1

    xt = t[...,None,None,None] * x + (1 - t[...,None,None,None]) * jax.random.normal(keys[1], shape=x.shape)  
    cosmot = t[...,None] * cosmo + (1 - t[...,None]) * jax.random.normal(keys[2], shape=cosmo.shape)

    model_vmap = jax.vmap(model, in_axes=(0,0,0,None))
    x_pred, cosmo_pred = model_vmap(xt, cosmot, t, False)

    # Denormalize cosmological parameters for display
    def denormalize(cosmo_norm):
        return cosmo_norm * THETA_STD + THETA_MEAN
    
    cosmo_denorm = denormalize(cosmo)
    cosmot_denorm = denormalize(cosmot)
    cosmo_pred_denorm = denormalize(cosmo_pred)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.5], 
                          hspace=0.3, wspace=0.2)
    
    # Plot images in first 5 columns
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
                    ax.set_ylabel('Noisy', fontsize=12, fontweight='bold')
            elif j == 2:
                ax.imshow(x_pred[0, ..., i], cmap='viridis')
                if i == 0:
                    ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
            ax.axis('off')
    
    # Add text information in the 6th column
    param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w0']
    
    for j in range(3):
        ax_text = fig.add_subplot(gs[j, 5])
        ax_text.axis('off')
        
        if j == 0:
            cosmo_vals = cosmo_denorm[0]
            title = 'Ground Truth\nCosmology'
        elif j == 1:
            cosmo_vals = cosmot_denorm[0]
            title = 'Noisy\nCosmology'
        else:
            cosmo_vals = cosmo_pred_denorm[0]
            title = 'Predicted\nCosmology'
        
        text_str = f'{title}\n' + '─' * 15 + '\n'
        for name, val in zip(param_names, cosmo_vals):
            text_str += f'{name:>4s}: {val:7.4f}\n'
        
        ax_text.text(0.1, 0.5, text_str, 
                    fontsize=10, 
                    family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    return fig


def denormalize_cosmo(cosmo_norm):
    """Denormalize cosmological parameters."""
    return cosmo_norm * THETA_STD + THETA_MEAN


def denormalize_fields(fields_norm):
    """Denormalize field data."""
    field_mean = FIELD_MEAN.reshape(1, 1, 1, -1)
    field_std = FIELD_STD.reshape(1, 1, 1, -1)
    return fields_norm * field_std + field_mean


def plot_samples(x_samples, cosmo_samples, n_samples=6, denormalize=True):
    """Plot generated samples with cosmological parameters.
    
    Args:
        x_samples: Generated field samples [batch, H, W, channels]
        cosmo_samples: Generated cosmology samples [batch, n_params]
        n_samples: Number of samples to plot (default 6)
        denormalize: Whether to denormalize fields for display
    """
    n_samples = min(n_samples, x_samples.shape[0])
    n_channels = x_samples.shape[-1]
    
    # Denormalize data for display
    cosmo_denorm = denormalize_cosmo(cosmo_samples)
    if denormalize:
        x_display = denormalize_fields(x_samples)
    else:
        x_display = x_samples
    
    # Create figure with one row per sample
    fig = plt.figure(figsize=(3 * n_channels, 3 * n_samples))
    gs = fig.add_gridspec(n_samples, n_channels, hspace=0.3, wspace=0.1)
    
    param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w0']
    
    for i in range(n_samples):
        # Create title with cosmological parameters
        cosmo_vals = cosmo_denorm[i]
        title_parts = [f'{name}={val:.3f}' for name, val in zip(param_names, cosmo_vals)]
        title = '  |  '.join(title_parts)
        
        for j in range(n_channels):
            ax = fig.add_subplot(gs[i, j])
            
            # Plot field channel
            im = ax.imshow(x_display[i, ..., j], cmap='viridis', aspect='auto')
            ax.axis('off')
            
            # Add column header for first row
            if i == 0:
                ax.set_title(f'Channel {j}', fontsize=10, fontweight='bold')
            
            # Add cosmology parameters as row title (left side of first column)
            if j == 0:
                ax.text(-0.05, 0.5, title, 
                       transform=ax.transAxes,
                       fontsize=8,
                       verticalalignment='center',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Generated Samples: Density Fields + Cosmological Parameters', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    return fig


def normalize_batch(batch):
    """Normalize a batch from the dataset."""
    theta_norm = (batch['theta'] - THETA_MEAN) / THETA_STD
    
    field_mean = FIELD_MEAN.reshape(1, 1, 1, -1)
    field_std = FIELD_STD.reshape(1, 1, 1, -1)
    # map_norm = (batch['map'] - field_mean) / field_std
    map_norm = batch['map'] * 100.
    
    return {'map': map_norm, 'theta': theta_norm}


def create_optimizer(cfg, total_steps):
    """Create optimizer with optional learning rate schedule"""
    
    if cfg['optimizer']['use_schedule']:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=cfg['optimizer']['schedule']['init_value'],
            peak_value=cfg['optimizer']['schedule']['peak_value'],
            warmup_steps=cfg['optimizer']['schedule']['warmup_steps'],
            decay_steps=total_steps,
            end_value=cfg['optimizer']['schedule']['end_value'],
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


def train(cfg):
    """Main training function"""
    
    # Enable mixed precision if requested
    if cfg['training']['use_mixed_precision']:
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
        print(f"Mixed precision enabled: {cfg['training']['precision']}")
    
    # Initialize wandb
    run = wandb.init(
        project=cfg['logging']['project'],
        entity=cfg['logging']['entity'],
        config=cfg,
    )
    
    # Save config to wandb run directory
    config_save_path = os.path.join(wandb.run.dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Create checkpoint directory
    # checkpoint_dir = os.path.abspath(cfg['checkpoint']['dir'])
    checkpoint_dir = os.path.abspath(os.path.join(wandb.run.dir, 'checkpoints'))    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_path_base = os.path.join(checkpoint_dir, cfg['model']['name'])

    # Load dataset
    dataset = load_from_disk(cfg['data']['dataset_path'])
    dataset = dataset.with_format("numpy")

    dataset = dataset.train_test_split(
        test_size=cfg['data']['val_split'], 
        seed=cfg['data']['shuffle_seed']
    )
    
    ds_train = dataset["train"]
    ds_val = dataset["test"]
    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples: {len(ds_val)}")

    # Model initialization
    model = JADE_B_16(
        rngs=nnx.Rngs(cfg['training']['seed']), 
        in_channels=cfg['model']['in_channels'], 
        input_size=cfg['model']['input_size'],
        patch_size=16
    )


    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree.leaves(params))
    print(f"Total parameters: {total:,}")

    # # EMA setup
    # if cfg['ema']['use_ema']:
    #     ema_params = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
    #     print(f"EMA initialized with decay: {cfg['ema']['decay']}")
    # else:
    #     ema_params = None
    #     print("EMA disabled")

    # EMA setup
    if cfg['ema']['use_ema']:
        ema_params = None  # <-- Start as None
        print(f"EMA will be initialized after first epoch")
    else:
        ema_params = None
        print("EMA disabled")


    # Calculate total training steps
    steps_per_epoch = len(ds_train) // cfg['training']['batch_size']
    total_steps = cfg['training']['num_epochs'] * steps_per_epoch
    print(f"Total training steps: {total_steps}")

    # Optimizer setup
    opt = create_optimizer(cfg, total_steps)
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    # Get weight function
    # weight_fn = get_weight_fn(cfg)

    def loss_fn(model, x, cosmo, key, train=True, return_components=False):

        keys = jax.random.split(key, 3)

        # Sample time
        if cfg['diffusion']['time_distribution'] == "logit":
            print("Using logit time distribution")
            
            mu = cfg['diffusion']['mu']
            sigma = cfg['diffusion']['sigma']
            s = (jax.random.normal(keys[0], shape=x.shape[:1]) + mu) * sigma
            t = jax.nn.sigmoid(s)

        elif cfg['diffusion']['time_distribution'] == "beta":
            t = jax.random.beta(
                keys[0], 
                a=cfg['diffusion']['beta_a'], 
                b=cfg['diffusion']['beta_b'], 
                shape=x.shape[:1]
            )
        elif cfg['diffusion']['time_distribution'] == "uniform":
            t = jax.random.uniform(keys[0], shape=x.shape[:1])
        else:
            raise ValueError(f"Unknown time distribution: {cfg['diffusion']['time_distribution']}")
        
        # sigma_t = sigma_fn(t, cfg)
        
        # Forward diffusion
        # z = sigma_t[...,None,None,None] * jax.random.normal(keys[1], shape=x.shape)
        # xt = x + z

        # zc = sigma_t[...,None] * jax.random.normal(keys[2], shape=cosmo.shape)
        # cosmot = cosmo + zc

        xt = t[...,None,None,None] * x + (1 - t[...,None,None,None]) * jax.random.normal(keys[1], shape=x.shape)  
        cosmot = t[...,None] * cosmo + (1 - t[...,None]) * jax.random.normal(keys[2], shape=cosmo.shape)


        model_vmap = jax.vmap(model, in_axes=(0,0,0,None))
        x_pred, cosmo_pred = model_vmap(xt, cosmot, t, train)

        # Compute losses
        loss_x = jnp.mean((x - x_pred)**2, axis=(-1,-2,-3))
        loss_cosmo = jnp.mean((cosmo - cosmo_pred)**2, axis=-1)
        
        # Apply time weighting and loss weighting
        # weights = weight_fn(t)
        if cfg["loss"]["type"]=="x-loss":
            total_loss = jnp.mean(
                (loss_x + cfg['loss']['lambda_cosmo'] * loss_cosmo)  # * weights
            )

        elif cfg["loss"]["type"]=="v-loss":
            vx = (x - xt) / jnp.clip((1 - t[...,None,None,None]), a_min=0.05)
            vx_pred = (x_pred - xt) / jnp.clip((1 - t[...,None,None,None]), a_min=0.05) 

            vcosmo = (cosmo - cosmot) / jnp.clip((1 - t[...,None]), a_min=0.05)
            vcosmo_pred = (cosmo_pred - cosmot) / jnp.clip((1 - t[...,None]), a_min=0.05)
            total_loss = jnp.sum((vx - vx_pred)**2, (-1,-2,-3)) + cfg['loss']['lambda_cosmo'] * jnp.sum(
              (vcosmo - vcosmo_pred)**2, (-1))
            total_loss = total_loss.mean()

        if return_components:
            return total_loss, (jnp.mean(loss_x), jnp.mean(loss_cosmo))
        else:
            return total_loss

    @nnx.jit
    def train_step(model, optimizer, x, cosmo, key):
        loss_fn_ = partial(
            loss_fn, x=x, cosmo=cosmo, key=key, 
            train=True, return_components=False
        )
        loss, grads = nnx.value_and_grad(loss_fn_)(model)
        
        optimizer.update(model, grads)

        return loss

    @jax.jit
    def update_ema(ema_params, model_params, decay):
        """Update EMA parameters"""
        return jax.tree.map(
            lambda ema, new: decay * ema + (1 - decay) * new,
            ema_params,
            model_params
        )

    key = jax.random.key(cfg['training']['seed'])
    best_val_loss = float('inf')
    step = 0

    for epoch in range(cfg['training']['num_epochs']):
        loader = ds_train.shuffle(seed=epoch).iter(
            batch_size=cfg['training']['batch_size'], 
            drop_last_batch=True
        )

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['training']['num_epochs']}"):

            batch = normalize_batch(batch)

            key, subkey = jax.random.split(key, 2)
            x = augment(batch["map"], jax.random.split(subkey, len(batch["map"])))
            cosmo = batch["theta"]

            key, subkey = jax.random.split(key, 2)
            loss = train_step(model, optimizer, x, cosmo, subkey)
            
            # # Update EMA
            # if cfg['ema']['use_ema']:
            #     current_params = nnx.state(model, nnx.Param)
            #     ema_params = update_ema(ema_params, current_params, cfg['ema']['decay'])
            
            # Initialize EMA after first epoch
            if cfg['ema']['use_ema'] and ema_params is None and epoch >= 1:
                ema_params = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
                print(f"EMA initialized from trained model at epoch {epoch}")
            
            # Update EMA (only if initialized)
            if cfg['ema']['use_ema'] and ema_params is not None:
                current_params = nnx.state(model, nnx.Param)
                ema_params = update_ema(ema_params, current_params, cfg['ema']['decay'])
            

            # Logging
            if step % cfg['logging']['log_every_n_steps'] == 0:
                run.log({"train/loss_total": loss, "train/epoch": epoch})
            
            step += 1

        
        # Validation
        if cfg['ema']['use_ema'] and ema_params is not None:
            original_params = nnx.state(model, nnx.Param)
            nnx.update(model, ema_params)
        
        losses = []
        losses_x = []
        losses_cosmo = []
        val_loader = ds_val.iter(
            batch_size=cfg['training']['batch_size'], 
            drop_last_batch=False
        )
        for batch in val_loader:
            batch = normalize_batch(batch)
            x_val = batch["map"]
            cosmo_val = batch["theta"]
            
            key, val_key = jax.random.split(key, 2)
            loss_val = partial(loss_fn, train=False, return_components=True)
            val_loss, (val_loss_x, val_loss_cosmo) = nnx.jit(loss_val)(
                model, x_val, cosmo_val, val_key,
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
              f"(Field: {val_loss_x:.4f}, Cosmo: {val_loss_cosmo:.4f})")

        # Visualization
        if (epoch + 1) % cfg['logging']['visualize_every_n_epochs'] == 0:
            fig = plot_denoiser(x_val, cosmo_val, model, key, cfg)
            tag = "denoiser_ema" if cfg['ema']['use_ema'] else "denoiser"
            wandb.log({tag: wandb.Image(fig)})
            plt.close(fig)
        
        if (epoch + 1) % 5 == 0:
            # Sample
            denoiser = Denoiser(model, cfg)
            x_samples, cosmo_samples = denoiser.generate(key, batch_size=6, x_shape=x_val.shape, cosmo_shape=cosmo_val.shape)
            fig = plot_samples(x_samples, cosmo_samples, n_samples=6)
            wandb.log({"samples": wandb.Image(fig)})
            plt.close(fig)

        # Restore original params if using EMA
        if cfg['ema']['use_ema'] and ema_params is not None:
            nnx.update(model, original_params)

        # Save checkpoints
        if (epoch + 1) % cfg['checkpoint']['save_every_n_epochs'] == 0:
            
            # Save EMA checkpoint (primary)
            if cfg['ema']['use_ema'] and ema_params is not None:
                checkpointer.save(
                    checkpoint_path_base + '_ema_latest', 
                    ema_params, 
                    force=True
                )
            
            # Save training checkpoint
            checkpointer.save(
                checkpoint_path_base + '_latest', 
                nnx.state(model, nnx.Param), 
                force=True
            )
            
            # Save best checkpoint
            if cfg['checkpoint']['keep_best'] and val_loss < best_val_loss:
                best_val_loss = val_loss
                if cfg['ema']['use_ema'] and ema_params is not None:
                    checkpointer.save(
                        checkpoint_path_base + '_ema_best', 
                        ema_params, 
                        force=True
                    )
                checkpointer.save(
                    checkpoint_path_base + '_best', 
                    nnx.state(model, nnx.Param), 
                    force=True
                )
                print(f"✓ Best model saved (val_loss: {val_loss:.4f})")
            
            print(f"✓ Checkpoints saved")

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the JADE model.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    def parse_config(cfg):
        """Recursively convert string numbers to actual numbers in config"""
        if isinstance(cfg, dict):
            return {k: parse_config(v) for k, v in cfg.items()}
        elif isinstance(cfg, list):
            return [parse_config(v) for v in cfg]
        elif isinstance(cfg, str):
            # Try to convert to number
            try:
                if '.' in cfg or 'e' in cfg.lower():
                    return float(cfg)
                else:
                    return int(cfg)
            except ValueError:
                return cfg
        else:
            return cfg


    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = parse_config(cfg) 

    # Print config
    print("="*50)
    print("Configuration:")
    print("="*50)
    import pprint
    pprint.pprint(cfg)
    print("="*50)
    
    # Run training
    train(cfg)
