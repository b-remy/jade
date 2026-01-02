import os
from typing import Any, Dict
import flax.nnx as nnx
import yaml
import pickle
import numpy as np

import itertools
from astropy import units as u
from lenstools.statistics import ConvergenceMap

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


def fill_lower_diag(array,nl):
    n = int(np.sqrt(len(array)*2))+1
    mask = np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n, nl))
    out[np.stack(mask,axis=1)] = array
    return out.T

def compute_ps(m_data):
    l_edges_kmap= np.linspace(300, 5000, 128)
    
    map_size = 5
    
    lis=[0,1,2,3,4]
    p_cross = []
    
    for i, j in itertools.combinations(lis, 2):
        ell, ps = ConvergenceMap(
            m_data[:,:,i], 
            angle=map_size*u.deg
        ).cross(
            ConvergenceMap(
                m_data[:,:,j], 
               angle=map_size*u.deg),
            l_edges=l_edges_kmap)
        p_cross.append(ps)
        
    ps_cross=np.array(p_cross)
    ps_cross = fill_lower_diag(ps_cross, 127)
    
    ps_auto=[]
    for i in range(5):
        ell, pi = ConvergenceMap(
            m_data[:,:,i], 
            angle=map_size*u.deg
        ).cross(ConvergenceMap(m_data[:,:,i], angle=map_size*u.deg),l_edges=l_edges_kmap)
        ps_auto.append(pi)
    ps_auto = np.array(ps_auto)

    return ell, ps_auto, ps_cross