from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import condition, seed, trace
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

np.complex = complex
np.float = float

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"


def get_samples(
    model,
    key,
    batch_size=64,
    score_type="density",
    thetas=None,
    with_noise=True,
):
    """Handling function sampling and computing the score from the model.

    Parameters
    ----------
    model : numpyro model
    key : PRNG Key
    batch_size : int, optional
        size of the batch to sample, by default 64
    score_type : str, optional
        'density' for nabla_theta log p(theta | y, z) or
        'conditional' for nabla_theta log p(y | z, theta), by default 'density'
    thetas : Array (batch_size, 2), optional
        thetas used to sample simulations or
        'None' sample thetas from the model, by default None
    with_noise : bool, optional
        add noise in simulations, by default True
        note: if no noise the score is only nabla_theta log p(theta, z)
        and log_prob log p(theta, z)

    Returns
    -------
    Array
        (log_prob, sample), score
    """

    params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]

    def log_prob_fn(theta, key):
        cond_model = seed(model, key)
        cond_model = condition(
            cond_model,
            {
                "omega_c": theta[0],
                "omega_b": theta[1],
                "sigma_8": theta[2],
                "h_0": theta[3],
                "n_s": theta[4],
                "w_0": theta[5],
            },
        )
        model_trace = trace(cond_model).get_trace()
        sample = {
            "theta": jnp.stack([model_trace[name]["value"] for name in params_name], axis=-1),
            "y": model_trace["y"]["value"],
            "z": model_trace["z"]["value"],
        }

        if score_type == "density":
            logp = 0
            for name in params_name:
                logp += model_trace[name]["fn"].log_prob(model_trace[name]["value"])
        elif score_type == "conditional":
            logp = 0

        if with_noise:
            logp += model_trace["y"]["fn"].log_prob(jax.lax.stop_gradient(model_trace["y"]["value"])).sum()
        logp += model_trace["z"]["fn"].log_prob(model_trace["z"]["value"]).sum()

        return logp, sample

    # Split the key by batch
    keys = jax.random.split(key, batch_size)

    # Sample theta from the model
    if thetas is None:

        @jax.vmap
        def get_params(key):
            model_trace = trace(seed(model, key)).get_trace()
            thetas = jnp.stack([model_trace[name]["value"] for name in params_name], axis=-1)
            return thetas

        thetas = get_params(keys)

    # return jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas, keys)
    return jax.vmap(log_prob_fn)(thetas, keys)
