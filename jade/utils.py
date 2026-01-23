import os
from typing import Any, Dict
import flax.nnx as nnx
import yaml
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jade.init import THETA_MEAN, THETA_STD, FIELD_MEAN, FIELD_STD, sigma_lsst
import itertools
from astropy import units as u
# from lenstools.statistics import ConvergenceMap

# Save model
def dump_model(cfg: Dict[str, Any], state: Dict[str, Any], name:str, path: str) -> None:
    """Saves model state to a file using pickle.

    Args:
        cfg: Configuration dictionary.
        state: Model state dictionary to save.
        file: Path to the file where the state will be saved.
    """

    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    with open(os.path.join(path, f'{name}_state.pkl'), 'wb') as f:
        pickle.dump(state, f)

# Load model
def load_model(path: str, name: str):
    """Loads model state from a file using pickle.

    Args:
        path: Path to the directory from which the state will be loaded.
    Returns:
        Model state dictionary loaded from the file.
    """
    with open(os.path.join(path, 'config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    with open(os.path.join(path, f'{name}_state.pkl'), 'rb') as f:
        state = pickle.load(f)
    
    return cfg, state

def denormalize_cosmo(cosmo_norm):
    """Denormalize cosmological parameters."""
    return cosmo_norm * THETA_STD + THETA_MEAN


def denormalize_fields(fields_norm):
    """Denormalize field data."""
    field_mean = FIELD_MEAN.reshape(1, 1, 1, -1)
    field_std = FIELD_STD.reshape(1, 1, 1, -1)
    return fields_norm * field_std + field_mean

def plot_denoiser(x, cosmo, model, key, cfg):
    """Visualization function"""
    keys = jax.random.split(key, 3)
    
    # if cfg['diffusion']['time_distribution'] == "logit":
    mu = cfg['diffusion']['mu']
    sigma = cfg['diffusion']['sigma']
    s = (jax.random.normal(keys[0], shape=x.shape[:1]) + mu) * sigma
    t = jax.nn.sigmoid(s)

    xt, cosmot = jax.vmap(model.forward_coupling, in_axes=(0,0,0,None))(x, cosmo, t, keys[1])

    cond = x + sigma_lsst * jax.random.normal(keys[2], shape=x.shape)

    model_vmap = jax.vmap(model.x_pred, in_axes=(0,0,0,0,None))
    x_pred, cosmo_pred = model_vmap(xt, cosmot, t, cond, False)

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


# def fill_lower_diag(array,nl):
#     n = int(np.sqrt(len(array)*2))+1
#     mask = np.arange(n)[:,None] > np.arange(n)
#     out = np.zeros((n,n, nl))
#     out[np.stack(mask,axis=1)] = array
#     return out.T

# def compute_ps(m_data):
#     l_edges_kmap= np.linspace(300, 5000, 128)
    
#     map_size = 5
    
#     lis=[0,1,2,3,4]
#     p_cross = []
    
#     for i, j in itertools.combinations(lis, 2):
#         ell, ps = ConvergenceMap(
#             m_data[:,:,i], 
#             angle=map_size*u.deg
#         ).cross(
#             ConvergenceMap(
#                 m_data[:,:,j], 
#                angle=map_size*u.deg),
#             l_edges=l_edges_kmap)
#         p_cross.append(ps)
        
#     ps_cross=np.array(p_cross)
#     ps_cross = fill_lower_diag(ps_cross, 127)
    
#     ps_auto=[]
#     for i in range(5):
#         ell, pi = ConvergenceMap(
#             m_data[:,:,i], 
#             angle=map_size*u.deg
#         ).cross(ConvergenceMap(m_data[:,:,i], angle=map_size*u.deg),l_edges=l_edges_kmap)
#         ps_auto.append(pi)
#     ps_auto = np.array(ps_auto)

#     return ell, ps_auto, ps_cross