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

from jade.nn import JiT_B_16  # Use the simple JiT model instead of JADE
from jade.init import FIELD_MEAN, FIELD_STD


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


def plot_denoiser(x, model, key, cfg):
    """Visualization function for field denoising only"""
    keys = jax.random.split(key, 2)
    
    # Sample timesteps
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
    
    # Forward diffusion (flow matching)
    xt = t[..., None, None, None] * x + (1 - t[..., None, None, None]) * jax.random.normal(keys[1], shape=x.shape)

    # Dummy class label (0 for unconditional)
    y = jnp.zeros(x.shape[0], dtype=jnp.int32)
    
    # Model prediction
    model_vmap = jax.vmap(model, in_axes=(0, 0, 0, None))
    x_pred = model_vmap(xt, t, y, False)

    # Plot: show 5 samples across 3 rows (ground truth, noisy, predicted)
    fig = plt.figure(figsize=(15, 9))
    
    for i in range(min(5, x.shape[0])):
        # Ground truth
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(x[i, ..., 0], cmap='viridis')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_title(f't={t[i]:.2f}')
        
        # Noisy
        ax = plt.subplot(3, 5, i + 6)
        ax.imshow(xt[i, ..., 0], cmap='viridis')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Noisy Input', fontsize=12, fontweight='bold')
        
        # Predicted
        ax = plt.subplot(3, 5, i + 11)
        ax.imshow(x_pred[i, ..., 0], cmap='viridis')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Denoised', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def normalize_batch(batch):
    """Normalize a batch from the dataset."""
    # Just normalize the maps
    map_norm = batch['map'] * 100.  # Simple scaling
    
    return {'map': map_norm}


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
    """Main training function - field denoising only"""
    
    # Enable mixed precision if requested
    if cfg['training']['use_mixed_precision']:
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
        print(f"Mixed precision enabled: {cfg['training']['precision']}")
    
    # Initialize wandb
    run = wandb.init(
        project=cfg['logging']['project'],
        entity=cfg['logging']['entity'],
        config=cfg,
        name=cfg.get('run_name', 'jit_field_only')
    )
    
    # Save config to wandb run directory
    config_save_path = os.path.join(wandb.run.dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Create checkpoint directory
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

    # Model initialization - Use JiT instead of JADE
    model = JiT_B_16(
        rngs=nnx.Rngs(cfg['training']['seed']), 
        in_channels=cfg['model']['in_channels'], 
        input_size=cfg['model']['input_size'],
        # patch_size=cfg['model']['patch_size'],
    )

    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree.leaves(params))
    print(f"Total parameters: {total:,}")

    # EMA setup
    if cfg['ema']['use_ema']:
        ema_params = None  # Start as None, initialize after first epoch
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

    def loss_fn(model, x, key, train=True, return_components=False):
        """Loss function for field denoising only"""
        
        keys = jax.random.split(key, 2)

        # Sample time
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
        elif cfg['diffusion']['time_distribution'] == "uniform":
            t = jax.random.uniform(keys[0], shape=x.shape[:1])
        else:
            raise ValueError(f"Unknown time distribution: {cfg['diffusion']['time_distribution']}")
        
        # Forward diffusion (flow matching formulation)
        xt = t[..., None, None, None] * x + (1 - t[..., None, None, None]) * jax.random.normal(keys[1], shape=x.shape)

        # Dummy class labels (0 = unconditional)
        y = jnp.zeros(x.shape[0], dtype=jnp.int32)
        
        # Model forward pass
        model_vmap = jax.vmap(model, in_axes=(0, 0, 0, None))
        x_pred = model_vmap(xt, t, y, train)

        # Compute loss
        if cfg["loss"]["type"] == "x-loss":
            # Direct prediction loss
            loss_x = jnp.mean((x - x_pred)**2, axis=(-1, -2, -3))
            total_loss = jnp.mean(loss_x)
        
        elif cfg["loss"]["type"] == "v-loss":
            # Velocity prediction loss
            vx = (x - xt) / jnp.clip((1 - t[..., None, None, None]), a_min=0.05)
            vx_pred = (x_pred - xt) / jnp.clip((1 - t[..., None, None, None]), a_min=0.05)
            loss_x = jnp.mean((vx - vx_pred)**2, axis=(-1, -2, -3))
            total_loss = jnp.mean(loss_x)
        
        else:
            raise ValueError(f"Unknown loss type: {cfg['loss']['type']}")

        if return_components:
            return total_loss, loss_x.mean()
        else:
            return total_loss

    @nnx.jit
    def train_step(model, optimizer, x, key):
        """Single training step"""
        loss_fn_ = partial(
            loss_fn, x=x, key=key, 
            train=True, return_components=False
        )
        loss, grads = nnx.value_and_grad(loss_fn_)(model)
        #optimizer.update(grads)
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

    # Training loop
    for epoch in range(cfg['training']['num_epochs']):
        loader = ds_train.shuffle(seed=epoch).iter(
            batch_size=cfg['training']['batch_size'], 
            drop_last_batch=True
        )

        epoch_losses = []
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['training']['num_epochs']}"):
            batch = normalize_batch(batch)

            # Augmentation
            key, subkey = jax.random.split(key, 2)
            x = augment(batch["map"], jax.random.split(subkey, len(batch["map"])))

            # Training step
            key, subkey = jax.random.split(key, 2)
            loss = train_step(model, optimizer, x, subkey)
            epoch_losses.append(float(loss))
            
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
                run.log({
                    "train/loss": loss,
                    "train/epoch": epoch,
                    "train/step": step
                })
            
            step += 1

        # Log epoch-level metrics
        epoch_loss_mean = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss_mean:.6f}")
        run.log({
            "train/epoch_loss": epoch_loss_mean,
            "epoch": epoch + 1
        })
        
        # Validation
        if cfg['ema']['use_ema'] and ema_params is not None:
            original_params = nnx.state(model, nnx.Param)
            nnx.update(model, ema_params)
        
        val_losses = []
        val_loader = ds_val.iter(
            batch_size=cfg['training']['batch_size'], 
            drop_last_batch=False
        )
        for batch in val_loader:
            batch = normalize_batch(batch)
            x_val = batch["map"]
            
            key, val_key = jax.random.split(key, 2)
            val_loss, val_loss_field = loss_fn(
                model, x_val, val_key, 
                train=False, return_components=True
            )
            val_losses.append(float(val_loss))

        val_loss = np.mean(val_losses)
        
        run.log({
            "val/loss": val_loss,
            "epoch": epoch + 1,
        })

        model_type = "EMA" if cfg['ema']['use_ema'] else "Standard"
        print(f"Epoch {epoch + 1}, Val Loss ({model_type}): {val_loss:.6f}")

        # Visualization
        if (epoch + 1) % cfg['logging']['visualize_every_n_epochs'] == 0:
            fig = plot_denoiser(x_val[:5], model, key, cfg)
            tag = "denoiser_ema" if cfg['ema']['use_ema'] and ema_params is not None else "denoiser"
            wandb.log({tag: wandb.Image(fig)})
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
                print(f"✓ Best model saved (val_loss: {val_loss:.6f})")
            
            print(f"✓ Checkpoints saved at epoch {epoch + 1}")

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JiT model on fields only.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/field_only.yaml",
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
