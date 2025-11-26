#!/usr/bin/env python

import os

import jax
import jax.numpy as jnp

from flax import nnx
import optax

import numpy as np

from jade.nn import JiT_B_16
from datasets import load_from_disk
from functools import partial

import matplotlib.pyplot as plt

# from utils import PATH, plot_ae_residuals, random_flip

import wandb

from tqdm import tqdm

def plot_denoiser(x, model, key):
    keys = jax.random.split(key, 2)
    
    t = jax.random.beta(keys[0], a=3, b=3, shape=x.shape[:1])
    sigma_t = sigma_fn(t)
    
    # forward diffusion
    z = sigma_t[...,None,None,None] * jax.random.normal(keys[1], shape=x.shape)
    xt = x + z

    model = jax.vmap(model, in_axes=(0,0,None,None))
    x_pred = model(xt, sigma_t, 0, False)  # call methods directly

    fig = plt.figure(figsize=(15, 9))  # Larger figure for 3x5 subplots
    for i in range(5):
        for j in range(3):
            plt.subplot(3, 5, i + 1 + j*5)
            if j == 0:
                plt.imshow(x[0, ..., i], cmap='viridis')
            elif j == 1:
                plt.imshow(xt[0, ..., i], cmap='viridis')
            elif j == 2:
                plt.imshow(x_pred[0, ..., i], cmap='viridis')
            plt.axis('off')  # Optional: hide axes for cleaner look
            
    # Log the figure to wandb

    return fig

def train():

    run = wandb.init(
        project="jade", 
        entity="b-remy"
        )

    # Load dataset
    dataset = load_from_disk("lensing_dataset")
    dataset = dataset.with_format("numpy")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    ds_train = dataset["train"]
    ds_val = dataset["test"]

    # Model initialization
    model = JiT_B_16(rngs=nnx.Rngs(0), in_channels=5, input_size=128)

    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree.leaves(params))
    print(f"Total parameters: {total:,}")

    # Optimizer setup
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    def sigma_fn(t, sigma_min=1e-2, sigma_max=1.):
        return sigma_min * (sigma_max / sigma_min) ** (t)

    weight_fn = lambda t: 1 / sigma_fn(t)**2 + 1

    # Start with a classical variance exploding SDE
    
    def loss_fn(model, x, key, train=True):

        keys = jax.random.split(key, 2)

        # sample time
        t = jax.random.beta(keys[0], a=3, b=3, shape=x.shape[:1])
        sigma_t = sigma_fn(t)
        
        # forward diffusion
        z = sigma_t[...,None,None,None] * jax.random.normal(keys[1], shape=x.shape)
        xt = x + z

        model = jax.vmap(model, in_axes=(0,0,None,None))
        x_pred = model(xt, sigma_t, 0, train)  # call methods directly

        # l2 loss
        loss = (x - x_pred)**2

        return jnp.mean(loss.sum((-1,-2,-3)) * weight_fn(t))

    @nnx.jit  # Automatic state management for JAX transforms.
    def train_step(model, optimizer, x, key):

        # VE SDE loss function
        # loss = loss_fn(model, x, key, train=True)
        loss_fn_ = partial(loss_fn, x=x, key=key, train=True)
        loss, grads = nnx.value_and_grad(loss_fn_)(model)
        optimizer.update(model, grads)  # in-place updates

        return loss


    num_epochs = 10
    batch_size = 256

    key = jax.random.key(0)

    for epoch in range(num_epochs):
        loader = ds_train.shuffle(seed=epoch).iter(batch_size=batch_size, drop_last_batch=True)

        for batch in tqdm(loader):
            
            x = batch["map"]
            # x = augmentation(x, key)
            # params, opt_state = update(params, opt_state, batch)
            keys, subkey = jax.random.split(key, 2)
            loss = train_step(model, optimizer, x, subkey)
            run.log({"loss_train": loss})

        
        # Validation
        losses = []
        val_loader = ds_val.iter(batch_size=batch_size, drop_last_batch=False)
        for batch in val_loader:
            x_val = batch["map"]
            #model_ = jax.vmap(model, in_axes=(0,0,None,None))
            val_loss = loss_fn(model, x_val, key, train=False)
            losses.append(val_loss)

        val_loss = np.mean(losses)
        run.log({"val_loss": val_loss})

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

        fig = plot_denoiser(x_val, model, key)
        wandb.log({"denoiser": wandb.Image(fig)})
        plt.close(fig)

if __name__ == "__main__":
    # runid = wandb.util.generate_id()
    # train(runid=runid)
    train()
