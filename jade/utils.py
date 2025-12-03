import os
from typing import Any, Dict
import flax.nnx as nnx
import orbax.checkpoint as ocp

# Orbax/NNX uses PyTree structures (dicts, tuples, etc. containing JAX arrays)
PyTree = Any 

def unwrap_value_keys(tree: PyTree) -> PyTree:
    """
    Recursively removes the explicit {'value': array} structure from parameters
    in a checkpoint, common when saving raw nnx.State.

    Args:
        tree: The PyTree (dict/nnx.State) structure loaded from the checkpoint.
              The structure is assumed to be similar to:
              {'layer_name': {'kernel': {'value': Array}, 'bias': {'value': Array}}}

    Returns:
        A new PyTree where any dictionary that only contains the key 'value'
        is replaced by its value (the Array).
        Resulting structure: {'layer_name': {'kernel': Array, 'bias': Array}}
    """
    if isinstance(tree, dict):
        # If this dict is exactly {'value': something}, unwrap it
        if 'value' in tree and len(tree) == 1:
            return tree['value']
        # Otherwise, recurse down through its items
        return {k: unwrap_value_keys(v) for k, v in tree.items()}
    return tree

def load_nnx_checkpoint(
    checkpoint_path: str,
    model: nnx.Module
) -> PyTree:
    """
    Loads an NNX checkpoint and automatically handles the common issue of
    the nested {'value': array} structure.

    Args:
        checkpointer: The initialized Orbax PyTreeCheckpointer instance.
        checkpoint_path: The file path to the checkpoint directory/file.
        model: The initialized nnx.Module instance. Used to check the structure.

    Returns:
        The cleaned PyTree of parameters ready to be loaded into the model
        using nnx.update(model, loaded_params).
    """

    checkpointer = ocp.PyTreeCheckpointer()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 1. Load the raw checkpoint structure
    print(f"Loading raw checkpoint from: {checkpoint_path}")
    raw_params = checkpointer.restore(checkpoint_path)

    # 2. Clean the parameters by unwrapping the {'value': ...} structure
    clean_params = unwrap_value_keys(raw_params)

    # Optional: Check for top-level wrapper keys (e.g., 'params')
    # If the model's top-level structure doesn't match the loaded structure, 
    # we need to check for a common wrapper like 'params'.
    model_state = nnx.state(model)
    model_keys = set(model_state.keys())
    
    if len(clean_params) == 1 and list(clean_params.keys())[0] not in model_keys:
        top_level_key = list(clean_params.keys())[0]
        # Check if the single top-level key matches the first key of the model state
        if top_level_key == list(model_keys)[0]:
            print(f"Warning: Checkpoint may have a mismatched top-level key: '{top_level_key}'. Returning raw clean_params.")
        else:
            print(f"Detected top-level wrapper key '{top_level_key}'. Unwrapping...")
            clean_params = clean_params[top_level_key]
    
    print("Checkpoint successfully loaded and prepared for nnx.update.")
    return clean_params
