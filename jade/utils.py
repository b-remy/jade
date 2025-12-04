import os
from typing import Any, Dict
import flax.nnx as nnx
import yaml
import pickle

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