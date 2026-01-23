#!/usr/bin/env python

import os
import argparse
import yaml
import gc

import jax
import jax.numpy as jnp

from flax import nnx
import optax

import dm_pix as pix

import numpy as np

from datasets import load_from_disk
from functools import partial

import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

# from jade.nn import JADE_B_16, JADE_L_16, JADE_M_16
from jade.nn_conditional import JADE_B_16
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, sigma_lsst
from jade.flow_conditional import Denoiser, FlowLoss
from jade.sampling_conditional import EulerSampler
from jade.utils import dump_model, load_model, denormalize_fields, denormalize_cosmo, plot_denoiser, plot_samples

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


@jax.jit
def normalize_batch(batch):
    """Normalize a batch from the dataset."""
    theta_norm = (batch['theta'] - THETA_MEAN) / THETA_STD
    
    field_mean = FIELD_MEAN.reshape(1, 1, 1, -1)
    field_std = FIELD_STD.reshape(1, 1, 1, -1)
    map_norm = (batch['map'] - field_mean) / field_std
    
    return {'map': map_norm, 'theta': theta_norm}

@jax.jit
def make_cond(x, key):
    cond = x + sigma_lsst * jax.random.normal(key, shape=x.shape)
    return cond

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
    
    # Print diffusion mode
    diffusion_mode = cfg['diffusion'].get('mode', 'linear')
    print(f"Diffusion mode: {diffusion_mode}")
    if diffusion_mode == 'variance_exploding' or diffusion_mode == 've':
        print(f"  σ_min: {cfg['diffusion']['sigma_min']}")
        print(f"  σ_max: {cfg['diffusion']['sigma_max']}")
        print(f"  Weight type: {cfg['diffusion']['weight_type']}")
    
    # Save config to wandb run directory
    config_save_path = os.path.join(wandb.run.dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Create checkpoint directory
    checkpoint_dir = os.path.abspath(os.path.join(wandb.run.dir, 'checkpoints'))    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_from_disk(cfg['data']['dataset_path'])
    dataset = dataset.with_format("numpy")

    # dataset = dataset.train_test_split(test_size=0.1)["test"]
    
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
        enable_cond_image=cfg["model"]["enable_cond_image"],
        cond_channels=cfg["model"]["cond_channels"],
        patch_size=16
    )
    # model = JADE_L_16(
    #     rngs=nnx.Rngs(cfg['training']['seed']), 
    #     in_channels=cfg['model']['in_channels'], 
    #     input_size=cfg['model']['input_size'],
    #     patch_size=16
    # )

    # model = JADE_M_16(
    #     rngs=nnx.Rngs(cfg['training']['seed']), 
    #     in_channels=cfg['model']['in_channels'], 
    #     input_size=cfg['model']['input_size'],
    #     patch_size=16
    # )

    model = Denoiser(model, cfg)

    if cfg['start_from_checkpoint']:
        print(f"Loading model from checkpoint: {cfg['params_path']}")

        # _, params = load_model(cfg['params_path'], f"{cfg['model']['name']}_latest")
        _, ema_params = load_model(cfg['params_path'], f"{cfg['model']['name']}_ema_latest")
        
        nnx.update(model, ema_params)

    else:
        params = nnx.state(model, nnx.Param)
        # EMA setup
        if cfg['ema']['use_ema']:
            ema_params = None  # <-- Start as None
            print(f"EMA will be initialized after first epoch")
        else:
            ema_params = None
            print("EMA disabled")
    
    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree.leaves(params))
    print(f"Total parameters: {total:,}")

    # Calculate total training steps
    steps_per_epoch = len(ds_train) // cfg['training']['batch_size']
    total_steps = cfg['training']['num_epochs'] * steps_per_epoch
    print(f"Total training steps: {total_steps}")

    # Optimizer setup
    opt = create_optimizer(cfg, total_steps)
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    loss_fn = FlowLoss(cfg)

    @nnx.jit
    def train_step(model, optimizer, x, cosmo, key, cond=None):
        (loss,(_, _)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model=model, x=x, cosmo=cosmo, cond=cond, key=key, lambda_cosmo=cfg['loss']['lambda_cosmo'], train=True
        )
        
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

    lap = 0 if not cfg['start_from_checkpoint'] else cfg['lap']
    num_epochs = cfg['training']['num_epochs']

    for epoch in range(num_epochs):
        loader = ds_train.shuffle(seed=lap*num_epochs + epoch).iter(
            batch_size=cfg['training']['batch_size'], 
            drop_last_batch=True
        )

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['training']['num_epochs']}"):

            batch = normalize_batch(batch)

            key, subkey = jax.random.split(key, 2)
            x = augment(batch["map"], jax.random.split(subkey, len(batch["map"])))
            cosmo = batch["theta"]
            
            key, subkey = jax.random.split(key, 2)
            # Noisy field condition
            if cfg["model"]["enable_cond_image"]:
                cond_field = make_cond(x, subkey)
            else:
                cond_field = None

            key, subkey = jax.random.split(key, 2)
            loss = train_step(model, optimizer, x, cosmo, subkey, cond=cond_field)
            
            # Initialize EMA after first epoch
            if cfg['ema']['use_ema'] and ema_params is None and epoch >= 1 and not cfg["start_from_checkpoint"]:
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
            
            if cfg["model"]["enable_cond_image"]:
                cond_val = make_cond(x_val, val_key)
            else:
                cond_val = None

            val_loss, (val_loss_x, val_loss_cosmo) = loss_fn(
                model=model, x=x_val, cosmo=cosmo_val, key=val_key, 
                lambda_cosmo=cfg['loss']['lambda_cosmo'], train=False,
                cond=cond_val,
                # return_components=True
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
        
            key, subkey = jax.random.split(key, 2)
            
            sampler = EulerSampler(model=model, num_steps=50)
            
            keys = jax.random.split(subkey, 3)
            x_0 = jax.random.normal(keys[0], shape=(6, 128, 128, 5))
            cosmo_0 = jax.random.normal(keys[1], shape=(6, 6))

            keys = jax.random.split(keys[2], 6)
            if cfg["model"]["enable_cond_image"]:
                cond_plot = make_cond(x_val[:6], keys[2])
            else:
                cond_plot = None

            x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, cond_plot, keys)

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
            
            # Save best checkpoint
            if cfg['checkpoint']['keep_best'] and val_loss < best_val_loss:
                best_val_loss = val_loss
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
