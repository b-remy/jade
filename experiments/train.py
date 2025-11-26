#!/usr/bin/env python

import os
import argparse

import jax
import jax.numpy as jnp

from flax import nnx
from flax.training import orbax_utils
import orbax.checkpoint as ocp
import optax

import numpy as np

from datasets import load_from_disk
from functools import partial

import matplotlib.pyplot as plt

import wandb

from tqdm import tqdm

from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD
from jade.nn import JADE_B_16

SIGMA_MIN = 1e-2
SIGMA_MAX = 100.

def sigma_fn(t, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX):
        return sigma_min * (sigma_max / sigma_min) ** (t)


def plot_denoiser(x, cosmo, model, key, theta_mean, theta_std):
    keys = jax.random.split(key, 3)
    
    t = jax.random.beta(keys[0], a=3, b=3, shape=x.shape[:1])
    sigma_t = sigma_fn(t)
    
    # forward diffusion
    z = sigma_t[...,None,None,None] * jax.random.normal(keys[1], shape=x.shape)
    xt = x + z

    zc = sigma_t[...,None] * jax.random.normal(keys[2], shape=cosmo.shape)
    cosmot = cosmo + zc

    model = jax.vmap(model, in_axes=(0,0,0,None))
    x_pred, cosmo_pred = model(xt, cosmot, sigma_t, False)  # call methods directly

    # Denormalize cosmological parameters for display
    def denormalize(cosmo_norm):
        return cosmo_norm * theta_std + theta_mean
    
    cosmo_denorm = denormalize(cosmo)
    cosmot_denorm = denormalize(cosmot)
    cosmo_pred_denorm = denormalize(cosmo_pred)

    fig = plt.figure(figsize=(18, 10))
    
    # Create gridspec for more control over layout
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
        
        # Create text string
        text_str = f'{title}\n' + '─' * 15 + '\n'
        for name, val in zip(param_names, cosmo_vals):
            text_str += f'{name:>4s}: {val:7.4f}\n'
        
        ax_text.text(0.1, 0.5, text_str, 
                    fontsize=10, 
                    family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    return fig


def has_no_nans_batch(examples):
    """Check batches for NaNs (faster)"""
    valid_samples = []
    for map_data, theta_data in zip(examples['map'], examples['theta']):
        map_valid = not np.isnan(map_data).any()
        theta_valid = not np.isnan(theta_data).any()
        valid_samples.append(map_valid and theta_valid)
    return valid_samples

def normalize_batch(batch):
    """Normalize a batch from the dataset."""
    # Normalize theta: [batch_size, 6]
    theta_norm = (batch['theta'] - THETA_MEAN) / THETA_STD
    
    # Normalize map: [batch_size, 128, 128, 5]
    # Reshape field stats for broadcasting: [1, 1, 1, 5]
    field_mean = FIELD_MEAN.reshape(1, 1, 1, -1)
    field_std = FIELD_STD.reshape(1, 1, 1, -1)
    map_norm = (batch['map'] - field_mean) / field_std
    
    return {'map': map_norm, 'theta': theta_norm}

def train(num_epochs, batch_size, checkpoint_dir='checkpoints', ema_decay=0.9999):

    run = wandb.init(
        project="jade", 
        entity="b-remy"
        )

    # Create checkpoint directory
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup Orbax checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

    # Load dataset
    dataset = load_from_disk("sbi_lens_lognormal")
    dataset = dataset.with_format("numpy")

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    ds_train = dataset["train"]
    ds_val = dataset["test"]
    print(ds_train)
    print(ds_val)

    # Model initialization
    model = JADE_B_16(rngs=nnx.Rngs(0), in_channels=5, input_size=128)

    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree.leaves(params))
    print(f"Total parameters: {total:,}")

    # Create EMA shadow parameters (copy of model parameters)
    ema_params = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
    print(f"EMA initialized with decay: {ema_decay}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-4,
        warmup_steps=1000,
        decay_steps=num_epochs * len(ds_train) // batch_size,
        end_value=1e-5
    )

    # Optimizer setup
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=1e-4
        ),
    )

    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    weight_fn = lambda t: 1 / sigma_fn(t)**2 + 1

    def loss_fn(model, x, cosmo, key, train=True, return_components=False):
        keys = jax.random.split(key, 3)

        # sample time
        t = jax.random.beta(keys[0], a=3, b=3, shape=x.shape[:1])
        # t = jax.random.uniform(keys[0], shape=x.shape[:1])
        
        sigma_t = sigma_fn(t)
        
        # forward diffusion
        z = sigma_t[...,None,None,None] * jax.random.normal(keys[1], shape=x.shape)
        xt = x + z

        zc = sigma_t[...,None] * jax.random.normal(keys[2], shape=cosmo.shape)
        cosmot = cosmo + zc

        model = jax.vmap(model, in_axes=(0,0,0,None))
        x_pred, cosmo_pred = model(xt, cosmot, sigma_t, train)  

        # Compute losses
        loss_x = jnp.mean((x - x_pred)**2, axis=(-1,-2,-3))
        loss_cosmo = jnp.mean((cosmo - cosmo_pred)**2, axis=-1)
        
        # Apply time weighting
        weights = weight_fn(t)
        total_loss = jnp.mean((loss_x + loss_cosmo) * weights)
        
        if return_components:
            return total_loss, (jnp.mean(loss_x), jnp.mean(loss_cosmo))
        else:
            return total_loss

    @nnx.jit
    def train_step(model, optimizer, x, cosmo, key):

        loss_fn_ = partial(loss_fn, x=x, cosmo=cosmo, key=key, train=True, return_components=False)
        loss, grads = nnx.value_and_grad(loss_fn_)(model)
        
        optimizer.update(model, grads)

        return loss

    def update_ema(ema_params, model_params, decay):
        """Update EMA parameters"""
        return jax.tree.map(
            lambda ema, new: decay * ema + (1 - decay) * new,
            ema_params,
            model_params
        )

    key = jax.random.key(0)

    for epoch in range(num_epochs):
        loader = ds_train.shuffle(seed=epoch).iter(batch_size=batch_size, drop_last_batch=True)

        for batch in tqdm(loader):

            batch = normalize_batch(batch)
            x = batch["map"]
            cosmo = batch["theta"]

            key, subkey = jax.random.split(key, 2)
            loss = train_step(model, optimizer, x, cosmo, subkey)
            
            current_params = nnx.state(model, nnx.Param)
            ema_params = update_ema(ema_params, current_params, ema_decay)

            run.log({
                "train/loss_total": loss,
            })

        
        # Validation

        # with the EMA weights
        original_params = nnx.state(model, nnx.Param)
        # Temporarily load EMA params into model for validation
        nnx.update(model, ema_params)

        losses = []
        losses_x = []
        losses_cosmo = []
        val_loader = ds_val.iter(batch_size=batch_size, drop_last_batch=False)
        for batch in val_loader:
            batch = normalize_batch(batch)
            x_val = batch["map"]
            cosmo_val = batch["theta"]
            val_loss, (val_loss_x, val_loss_cosmo) = loss_fn(
                model, x_val, cosmo_val, key, train=False, return_components=True
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
            "val/loss_cosmo": val_loss_cosmo
        })

        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f} (Field: {val_loss_x:.4f}, Cosmo: {val_loss_cosmo:.4f})")

        # Save checkpoint (overwrites previous)
        abstract_state = nnx.state(model)
        checkpointer.save(checkpoint_path, abstract_state, force=True)
        print(f"Checkpoint saved to {checkpoint_path}")

        fig = plot_denoiser(x_val, cosmo_val, model, key, THETA_MEAN, THETA_STD)
        wandb.log({"denoiser": wandb.Image(fig)})
        plt.close(fig)

        # Restore original model params
        nnx.update(model, original_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the JADE model.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    args = parser.parse_args()

    train(num_epochs=args.num_epochs, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir)