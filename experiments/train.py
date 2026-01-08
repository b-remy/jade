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

from jade.nn import JADE_B_16, JADE_L_16, JADE_M_16 
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.flow import Denoiser, FlowLoss
from jade.sampling import EulerSampler
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

    # @nnx.jit
    # def loss_fn(model, x, cosmo, key):

    #     keys = jax.random.split(key, 3)

    #     # Sample time
    #     if cfg['diffusion']['time_distribution'] == "logit":
            
    #         mu = cfg['diffusion']['mu']
    #         sigma = cfg['diffusion']['sigma']
    #         s = (jax.random.normal(keys[0], shape=x.shape[:1]) + mu) * sigma
    #         t = jax.nn.sigmoid(s)

    #     elif cfg['diffusion']['time_distribution'] == "beta":
    #         t = jax.random.beta(
    #             keys[0], 
    #             a=cfg['diffusion']['beta_a'], 
    #             b=cfg['diffusion']['beta_b'], 
    #             shape=x.shape[:1]
    #         )
    #     elif cfg['diffusion']['time_distribution'] == "uniform":
    #         t = jax.random.uniform(keys[0], shape=x.shape[:1])
    #     else:
    #         raise ValueError(f"Unknown time distribution: {cfg['diffusion']['time_distribution']}")
        
    #     # Get diffusion mode
    #     diffusion_mode = cfg['diffusion'].get('mode', 'linear')
        
    #     if diffusion_mode == 'linear':
    #         # Linear interpolant (current implementation)
    #         xt = t[...,None,None,None] * x + (1 - t[...,None,None,None]) * jax.random.normal(keys[1], shape=x.shape)  
    #         cosmot = t[...,None] * cosmo + (1 - t[...,None]) * jax.random.normal(keys[2], shape=cosmo.shape)
    #         sigma_t = t
            
    #     elif diffusion_mode == 'variance_exploding' or diffusion_mode == 've':
    #         # Variance exploding: xt = x + sigma_t * z
    #         sigma_t = sigma_fn(t, cfg)
            
    #         z = jax.random.normal(keys[1], shape=x.shape)
    #         xt = x + sigma_t[...,None,None,None] * z
            
    #         zc = jax.random.normal(keys[2], shape=cosmo.shape)
    #         cosmot = cosmo + sigma_t[...,None] * zc
            
    #     else:
    #         raise ValueError(f"Unknown diffusion mode: {diffusion_mode}. Must be 'linear' or 'variance_exploding'")

    #     model_vmap = jax.vmap(model, in_axes=(0,0,0,None))
    #     x_pred, cosmo_pred = model_vmap(xt, cosmot, sigma_t, True)

    #     # Compute losses based on diffusion mode and loss type
    #     if diffusion_mode == 'linear':
    #         # Linear interpolant losses
    #         loss_x = jnp.mean((x - x_pred)**2, axis=(-1,-2,-3))
    #         loss_cosmo = jnp.mean((cosmo - cosmo_pred)**2, axis=-1)
            
    #         if cfg["loss"]["type"]=="x-loss":
    #             total_loss = jnp.mean(
    #                 (loss_x + cfg['loss']['lambda_cosmo'] * loss_cosmo)
    #             )

    #         elif cfg["loss"]["type"]=="v-loss":
    #             vx = (x - xt) / jnp.clip((1 - t[...,None,None,None]), a_min=0.05)
    #             vx_pred = (x_pred - xt) / jnp.clip((1 - t[...,None,None,None]), a_min=0.05) 

    #             vcosmo = (cosmo - cosmot) / jnp.clip((1 - t[...,None]), a_min=0.05)
    #             vcosmo_pred = (cosmo_pred - cosmot) / jnp.clip((1 - t[...,None]), a_min=0.05)
    #             total_loss = jnp.mean((vx - vx_pred)**2, (-1,-2,-3)) + cfg['loss']['lambda_cosmo'] * jnp.mean(
    #               (vcosmo - vcosmo_pred)**2, (-1))
    #             total_loss = total_loss.mean()
    #         else:
    #             raise ValueError(f"Unknown loss type for linear mode: {cfg['loss']['type']}")
                
    #     elif diffusion_mode == 'variance_exploding' or diffusion_mode == 've':
    #         # Variance exploding losses
    #         sigma_t = sigma_fn(t, cfg)
    #         weights = get_weight_fn(cfg)(t)
            
    #         if cfg["loss"]["type"]=="x-loss":
    #             # Direct denoising objective
    #             loss_x = jnp.mean((x - x_pred)**2, axis=(-1,-2,-3))
    #             loss_cosmo = jnp.mean((cosmo - cosmo_pred)**2, axis=-1)
    #             total_loss = jnp.mean(
    #                 weights * (loss_x + cfg['loss']['lambda_cosmo'] * loss_cosmo)
    #             )
                
    #         # elif cfg["loss"]["type"]=="eps-loss":
    #         #     # Noise prediction objective: predict the noise z
    #         #     # Ground truth noise
    #         #     z_true = (xt - x) / sigma_t[...,None,None,None]
    #         #     zc_true = (cosmot - cosmo) / sigma_t[...,None]
                
    #         #     # Predicted noise from model output
    #         #     z_pred = (xt - x_pred) / sigma_t[...,None,None,None]
    #         #     zc_pred = (cosmot - cosmo_pred) / sigma_t[...,None]
                
    #         #     loss_x = jnp.mean((z_true - z_pred)**2, axis=(-1,-2,-3))
    #         #     loss_cosmo = jnp.mean((zc_true - zc_pred)**2, axis=-1)
    #         #     total_loss = jnp.mean(
    #         #         weights * (loss_x + cfg['loss']['lambda_cosmo'] * loss_cosmo)
    #         #     )
    #         else:
    #             raise ValueError(f"Unknown loss type for VE mode: {cfg['loss']['type']}")
            
    #         # Store unweighted losses for logging
    #         loss_x = jnp.mean((x - x_pred)**2, axis=(-1,-2,-3))
    #         loss_cosmo = jnp.mean((cosmo - cosmo_pred)**2, axis=-1)

    #     # if return_components:
    #     return total_loss, (jnp.mean(loss_x), jnp.mean(loss_cosmo))
    #     # else:
    #         # return total_loss

    loss_fn = FlowLoss(cfg)

    @nnx.jit
    def train_step(model, optimizer, x, cosmo, key):
        # loss_fn_ = partial(
        #     ,
        # )
        (loss,(_, _)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model=model, x=x, cosmo=cosmo, key=key, lambda_cosmo=cfg['loss']['lambda_cosmo'], train=True
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
            loss = train_step(model, optimizer, x, cosmo, subkey)
            
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
            
            val_loss, (val_loss_x, val_loss_cosmo) = loss_fn(
                model=model, x=x_val, cosmo=cosmo_val, key=val_key, 
                lambda_cosmo=cfg['loss']['lambda_cosmo'], train=False
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
        
        # if (epoch + 1) % 5 == 0:
            # Sample
            
            key, subkey = jax.random.split(key, 2)
            
            sampler = EulerSampler(model=model, num_steps=50)
            
            keys = jax.random.split(subkey, 3)
            x_0 = jax.random.normal(keys[0], shape=(6, 128, 128, 5))
            cosmo_0 = jax.random.normal(keys[1], shape=(6, 6))

            keys = jax.random.split(keys[2], 6)
            x_samples, cosmo_samples = jax.vmap(sampler)(x_0, cosmo_0, keys)

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