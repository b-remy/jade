import os
import argparse
import yaml

import jax
import jax.numpy as jnp

from flax import nnx
import orbax.checkpoint as ocp

import matplotlib.pyplot as plt
import numpy as np

from datasets import load_from_disk

import yaml
import pickle

import jax_cosmo as jc
import sbi_lens

import numpyro
import numpyro.distributions as dist
from numpyro import sample
from numpyro.handlers import condition, reparam, seed, trace

from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from sbi_lens.simulator.utils import get_reference_sample_posterior_full_field

from functools import partial

def main():

    save_dir = "./mcmc_log_normal"
    os.makedirs(save_dir, exist_ok=True)

    # sample at Planck15 fiducial cosmology
    cosmo = jc.parameters.Planck15()

    key = jax.random.key(0)    

    # generate mocked observation
    sigma_e = config_lsst_y_10.sigma_e
    gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
    nbins = config_lsst_y_10.nbins
    a = config_lsst_y_10.a
    b = config_lsst_y_10.b
    z0 = config_lsst_y_10.z0
    N = 128
    map_size = 5
    with_noise = True

    model_log_normal = partial(
        lensingLogNormal,
        N=N,
        map_size=map_size,
        gal_per_arcmin2=gals_per_arcmin2,
        sigma_e=sigma_e,
        nbins=nbins,
        a=a,
        b=b,
        z0=z0,
        model_type='lognormal',
        lognormal_shifts='LSSTY10',
        with_noise=with_noise,
        )

    cond_model = seed(model_log_normal, key)
    cond_model = condition(
        cond_model,
        {
            "omega_c": cosmo.Omega_c,
            "omega_b": cosmo.Omega_b,
            "sigma_8": cosmo.sigma8,
            "h_0": cosmo.h,
            "n_s": cosmo.n_s,
            "w_0": cosmo.w0,
        },
    )

    params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]

    model_trace = trace(cond_model).get_trace()
    sample = {
        "theta": jnp.stack(
            [model_trace[name]["value"] for name in params_name], axis=-1
        ),
        "y": model_trace["y"]["value"],
    }

    with open(os.path.join(save_dir, "mcmc_log_obs_truth.pkl"), "wb") as f:
        pickle.dump(sample, f)
    
    obs = sample["y"]
    
    # get reference posterior samples

    # initialize from the truth
    init_values = {k: model_trace[k]['value'] for k in ['z', 'omega_c', 'sigma_8', 'omega_b', 'h_0', 'n_s', 'w_0']}

    samples_mcmc = get_reference_sample_posterior_full_field(
        run_mcmc=True,
        N = 128,
        map_size=5.,
        model=model_log_normal,
        m_data=obs,
        num_results=3_000,
        num_warmup=500,
        nb_loop=1,
        init_strat=numpyro.infer.init_to_value(values=init_values),
        num_chains=10,
        chain_method="vectorized",
        key=key
    )

    with open(os.path.join(save_dir, "mcmc_log_posterior_samples.pkl"), "wb") as f:
        pickle.dump(samples_mcmc, f)

if __name__ == "__main__":
    main()
